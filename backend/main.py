"""FastAPI Server Entrypoint for Electro-Thermal PACS Workstation 2.0."""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncIterator
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes_kspace import gpu_session, router as kspace_router
from .api.ws_training import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initializes multi-coil k-space and Pennes Bioheat solver on the RTX 4060 Ti at startup."""
    gpu_session.initialize_default()
    yield
    if gpu_session.trainer is not None:
        gpu_session.trainer.stop_requested = True
    gpu_session.last_maps.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Electro-Thermal PACS Workstation 2.0 | cr-PINN + Pennes Bioheat + GradNorm",
    description="Second-generation Multi-Physics PyTorch + FastAPI backend for Viscosity-Free Convection-Reaction EPT and Pennes Bioheat Dosimetry.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kspace_router)
app.include_router(ws_router)


@app.get("/health")
async def health_check() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "version": "2.0.0",
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
