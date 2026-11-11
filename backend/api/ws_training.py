"""WebSocket Endpoint (/ws/train) for Real-Time GPU Electro-Thermal PINN Training and Telemetry."""

from __future__ import annotations
import asyncio
from typing import Any, Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.training.trainer import ElectroThermalPINNTrainer
from ..schemas import EpochTelemetryPayload, TrainingStartConfig
from .routes_kspace import gpu_session, tensor_to_f32_b64

router = APIRouter(tags=["websocket"])


def _build_trainer_config(cfg: TrainingStartConfig) -> Dict[str, Any]:
    """Builds configuration dictionary for ElectroThermalPINNTrainer."""
    if gpu_session.dataset is None:
        gpu_session.initialize_default()
    assert gpu_session.dataset is not None

    return {
        "physics": {
            "b0_tesla": cfg.b0_tesla,
            "larmor_freq_mhz": round(42.58 * cfg.b0_tesla, 2),
            "fov_x_m": gpu_session.dataset.fov_x_m,
            "fov_y_m": gpu_session.dataset.fov_y_m,
            "epsilon_b1": 1e-3,
            "epsilon_sigma": 1e-3,
            "use_convection_reaction": cfg.use_convection_reaction,
            "exposure_duration_min": cfg.exposure_duration_min,
            "tumor_perfusion_ratio": cfg.tumor_perfusion_ratio,
        },
        "data": {
            "mask_type": cfg.mask_type,
            "acceleration": cfg.acceleration,
            "acs_lines": cfg.acs_lines,
        },
        "model": {
            "omega_anatomy": cfg.omega_anatomy,
            "omega_coil": cfg.omega_coil,
            "omega_ept": cfg.omega_ept,
            "hidden_features_anatomy": 128,
            "hidden_layers_anatomy": 4,
            "hidden_features_coil": 64,
            "hidden_layers_coil": 3,
            "hidden_features_ept": 96,
            "hidden_layers_ept": 3,
            "sigma_bounds": [0.01, 3.0],
            "eps_r_bounds": [1.0, 100.0],
        },
        "training": {
            "epochs": cfg.epochs,
            "warmup_epochs": cfg.warmup_epochs,
            "lr_anatomy": cfg.lr_anatomy,
            "lr_coil": cfg.lr_coil,
            "lr_ept": cfg.lr_ept,
            "lambda_kspace": cfg.lambda_kspace,
            "lambda_ept_sigma": cfg.lambda_ept_sigma,
            "lambda_ept_eps": cfg.lambda_ept_eps,
            "lambda_coil_smooth": cfg.lambda_coil_smooth,
            "lambda_harmonic_b0": cfg.lambda_harmonic_b0,
            "use_gradnorm": cfg.use_gradnorm,
            "gradnorm_update_every": 5,
            "gradnorm_ema_alpha": 0.85,
            "telemetry_interval": cfg.telemetry_interval,
        },
    }


@router.websocket("/ws/train")
async def websocket_training_endpoint(websocket: WebSocket) -> None:
    """Bidirectional WebSocket streaming live GPU electro-thermal tensors, 1D profiles, and GradNorm weights."""
    await websocket.accept()
    training_task: Optional[asyncio.Task[None]] = None

    async def run_training_loop(cfg: TrainingStartConfig) -> None:
        try:
            gpu_session.update_mask_and_b0(
                mask_type=cfg.mask_type,
                acceleration=cfg.acceleration,
                acs_lines=cfg.acs_lines,
                b0_tesla=cfg.b0_tesla,
                exposure_duration_min=cfg.exposure_duration_min,
                tumor_perfusion_ratio=cfg.tumor_perfusion_ratio,
            )
            assert gpu_session.dataset is not None
            trainer_cfg = _build_trainer_config(cfg)
            trainer = ElectroThermalPINNTrainer(
                dataset=gpu_session.dataset,
                config=trainer_cfg,
                device=gpu_session.device,
            )
            gpu_session.trainer = trainer

            total_epochs = cfg.epochs
            interval = max(1, cfg.telemetry_interval)

            for epoch_idx in range(total_epochs):
                if trainer.stop_requested:
                    break

                metrics, maps = await asyncio.to_thread(trainer.step_epoch, epoch_idx)
                gpu_session.last_maps = maps

                include_maps = (
                    (epoch_idx + 1) == 1
                    or (epoch_idx + 1) == cfg.warmup_epochs
                    or (epoch_idx + 1) == cfg.warmup_epochs + 1
                    or (epoch_idx + 1) % interval == 0
                    or (epoch_idx + 1) == total_epochs
                )

                maps_payload: Optional[Dict[str, str]] = None
                if include_maps:
                    maps_payload = {
                        "m_mag": tensor_to_f32_b64(maps["m_mag"]),
                        "sigma": tensor_to_f32_b64(maps["sigma"]),
                        "sigma_helmholtz": tensor_to_f32_b64(maps["sigma_helmholtz"]),
                        "eps_r": tensor_to_f32_b64(maps["eps_r"]),
                        "sar": tensor_to_f32_b64(maps["sar"]),
                        "temperature_c": tensor_to_f32_b64(maps["temperature_c"]),
                        "cem43_min": tensor_to_f32_b64(maps["cem43_min"]),
                        "phi_tr": tensor_to_f32_b64(maps["phi_tr"]),
                        "phi_b0": tensor_to_f32_b64(maps["phi_b0"]),
                    }

                is_last = (epoch_idx + 1) == total_epochs and not trainer.stop_requested
                payload = EpochTelemetryPayload(
                    type="completed" if is_last else "epoch",
                    epoch=metrics.epoch,
                    total_epochs=total_epochs,
                    total_loss=metrics.total_loss,
                    loss_kspace=metrics.loss_kspace,
                    loss_ept=metrics.loss_ept,
                    loss_coil=metrics.loss_coil,
                    loss_harmonic=metrics.loss_harmonic,
                    psnr_db=metrics.psnr_db,
                    ssim=metrics.ssim,
                    nrmse_sigma=metrics.nrmse_sigma,
                    nrmse_eps=metrics.nrmse_eps,
                    tumor_sigma_mean=metrics.sigma_tumor_mean,
                    ventricle_sigma_mean=metrics.sigma_ventricle_mean,
                    max_temp_c=metrics.max_temp_c,
                    max_cem43_min=metrics.max_cem43_min,
                    iec_safety_status=str(maps["iec_safety_status"]),
                    lambda_ept=metrics.lambda_ept,
                    lambda_harm=metrics.lambda_harm,
                    ept_active=metrics.ept_active,
                    elapsed_ms=metrics.elapsed_ms,
                    thermal_timeline=maps["thermal_timeline"] if include_maps else None,
                    profile_1d=maps["profile_1d"] if include_maps else None,
                    maps_f32_b64=maps_payload,
                )
                await websocket.send_json(payload.model_dump())
                await asyncio.sleep(0.005)

            if trainer.stop_requested and trainer.history:
                last_m = trainer.history[-1]
                await websocket.send_json(
                    EpochTelemetryPayload(
                        type="stopped",
                        epoch=last_m.epoch,
                        total_epochs=total_epochs,
                        total_loss=last_m.total_loss,
                        loss_kspace=last_m.loss_kspace,
                        loss_ept=last_m.loss_ept,
                        loss_coil=last_m.loss_coil,
                        loss_harmonic=last_m.loss_harmonic,
                        psnr_db=last_m.psnr_db,
                        ssim=last_m.ssim,
                        nrmse_sigma=last_m.nrmse_sigma,
                        nrmse_eps=last_m.nrmse_eps,
                        tumor_sigma_mean=last_m.sigma_tumor_mean,
                        ventricle_sigma_mean=last_m.sigma_ventricle_mean,
                        max_temp_c=last_m.max_temp_c,
                        max_cem43_min=last_m.max_cem43_min,
                        iec_safety_status=str(gpu_session.last_maps.get("iec_safety_status", "STOPPED")),
                        lambda_ept=last_m.lambda_ept,
                        lambda_harm=last_m.lambda_harm,
                        ept_active=last_m.ept_active,
                        elapsed_ms=last_m.elapsed_ms,
                        message="Training stopped by user.",
                    ).model_dump()
                )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            await websocket.send_json(
                {
                    "type": "error",
                    "epoch": 0,
                    "total_epochs": cfg.epochs,
                    "total_loss": 0.0,
                    "loss_kspace": 0.0,
                    "loss_ept": 0.0,
                    "loss_coil": 0.0,
                    "loss_harmonic": 0.0,
                    "psnr_db": 0.0,
                    "ssim": 0.0,
                    "nrmse_sigma": 0.0,
                    "nrmse_eps": 0.0,
                    "tumor_sigma_mean": 0.0,
                    "ventricle_sigma_mean": 0.0,
                    "max_temp_c": 37.0,
                    "max_cem43_min": 0.0,
                    "iec_safety_status": "ERROR",
                    "lambda_ept": 0.0,
                    "lambda_harm": 0.0,
                    "ept_active": False,
                    "elapsed_ms": 0.0,
                    "message": str(exc),
                }
            )

    try:
        while True:
            data = await websocket.receive_json()
            action = str(data.get("action", "")).lower()

            if action == "start":
                if training_task is not None and not training_task.done():
                    if gpu_session.trainer is not None:
                        gpu_session.trainer.stop_requested = True
                    training_task.cancel()
                    await asyncio.sleep(0.05)

                raw_cfg = data.get("config", {})
                cfg = TrainingStartConfig(**raw_cfg)
                training_task = asyncio.create_task(run_training_loop(cfg))

            elif action == "stop":
                if gpu_session.trainer is not None:
                    gpu_session.trainer.stop_requested = True
    except WebSocketDisconnect:
        if gpu_session.trainer is not None:
            gpu_session.trainer.stop_requested = True
        if training_task is not None and not training_task.done():
            training_task.cancel()
