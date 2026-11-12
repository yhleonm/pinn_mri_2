import React, { useRef } from "react";
import {
  Activity,
  Cpu,
  Database,
  Play,
  Sliders,
  Square,
  Upload,
  Zap,
} from "lucide-react";
import {
  AccelerationFactor,
  B0FieldStrength,
  TrajectoryType,
  TrainingConfigRequest,
} from "../types/mri";

interface ControlSidebarProps {
  config: TrainingConfigRequest;
  onChangeConfig: (next: Partial<TrainingConfigRequest>) => void;
  isTraining: boolean;
  onStartTraining: () => void;
  onStopTraining: () => void;
  onLoadSampleH5: (sampleName: string) => void;
  onLoadUcBerkeley: () => void;
  onUploadH5: (file: File) => void;
  datasetTitle: string;
}

export const ControlSidebar: React.FC<ControlSidebarProps> = ({
  config,
  onChangeConfig,
  isTraining,
  onStartTraining,
  onStopTraining,
  onLoadSampleH5,
  onLoadUcBerkeley,
  onUploadH5,
  datasetTitle,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  return (
    <aside className="w-full lg:w-80 shrink-0 bg-slate-900/95 border border-slate-800 rounded-xl p-4 space-y-4 shadow-xl">
      {/* Station Identity */}
      <div className="border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[11px] font-mono uppercase tracking-wider text-emerald-400">
            ElectroThermalMRIPINN v2.0
          </span>
        </div>
        <h2 className="text-sm font-bold text-slate-100 mt-1">
          Control Multi-Física (cr-EPT + Pennes)
        </h2>
        <p className="text-[11px] text-slate-400 mt-0.5 truncate" title={datasetTitle}>
          Activo: {datasetTitle}
        </p>
      </div>

      {/* Dataset Acquisition Loader */}
      <div className="space-y-2">
        <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Database className="w-3.5 h-3.5 text-cyan-400" />
          1. Cohorte / Adquisición k-Space
        </label>
        <div className="grid grid-cols-2 gap-1.5">
          <button
            type="button"
            disabled={isTraining}
            onClick={() => onLoadSampleH5("brain_tumor")}
            className="text-[11px] bg-slate-800/90 hover:bg-slate-700 text-cyan-200 border border-slate-700 rounded-lg px-2.5 py-1.5 text-left transition disabled:opacity-50"
          >
            🧠 Glioma 3T (.h5)
          </button>
          <button
            type="button"
            disabled={isTraining}
            onClick={() => onLoadSampleH5("knee")}
            className="text-[11px] bg-slate-800/90 hover:bg-slate-700 text-teal-200 border border-slate-700 rounded-lg px-2.5 py-1.5 text-left transition disabled:opacity-50"
          >
            🦵 Rodilla 8ch (.h5)
          </button>
          <button
            type="button"
            disabled={isTraining}
            onClick={() => onLoadSampleH5("cardiac")}
            className="text-[11px] bg-slate-800/90 hover:bg-slate-700 text-rose-200 border border-slate-700 rounded-lg px-2.5 py-1.5 text-left transition disabled:opacity-50"
          >
            🫀 Cardíaco 6ch (.h5)
          </button>
          <button
            type="button"
            disabled={isTraining}
            onClick={() => onLoadSampleH5("torso")}
            className="text-[11px] bg-slate-800/90 hover:bg-slate-700 text-indigo-200 border border-slate-700 rounded-lg px-2.5 py-1.5 text-left transition disabled:opacity-50"
          >
            🫁 Torso 8ch (.h5)
          </button>
          <button
            type="button"
            disabled={isTraining}
            onClick={onLoadUcBerkeley}
            className="text-[11px] bg-slate-800/90 hover:bg-slate-700 text-emerald-200 border border-slate-700 rounded-lg px-2.5 py-1.5 text-left transition disabled:opacity-50"
          >
            🏥 UCB Cerebro 8ch
          </button>
          <button
            type="button"
            disabled={isTraining}
            onClick={() => onLoadSampleH5("shepp_logan")}
            className="text-[11px] bg-slate-800/90 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg px-2.5 py-1.5 text-left transition disabled:opacity-50"
          >
            🎯 Shepp-Logan (.h5)
          </button>
          <button
            type="button"
            disabled={isTraining}
            onClick={() => fileInputRef.current?.click()}
            className="col-span-2 text-[11px] bg-slate-800/90 hover:bg-slate-700 text-amber-200 border border-slate-700 rounded-lg px-2.5 py-1.5 flex items-center justify-center gap-1 transition disabled:opacity-50"
          >
            <Upload className="w-3 h-3" /> Subir archivo .h5 externo
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".h5,.hdf5"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onUploadH5(f);
            }}
          />
        </div>
      </div>

      {/* B0 Field Strength */}
      <div className="space-y-1.5">
        <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Zap className="w-3.5 h-3.5 text-amber-400" />
          2. Campo Magnético Estático B₀ (Dispersión Cole-Cole)
        </label>
        <div className="grid grid-cols-3 gap-1.5">
          {([1.5, 3.0, 7.0] as B0FieldStrength[]).map((b0) => (
            <button
              key={b0}
              type="button"
              disabled={isTraining}
              onClick={() => onChangeConfig({ b0_tesla: b0 })}
              className={`py-1.5 rounded-lg text-xs font-mono font-semibold border transition ${
                config.b0_tesla === b0
                  ? "bg-cyan-500/20 border-cyan-400 text-cyan-200 shadow-sm"
                  : "bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200"
              }`}
            >
              {b0.toFixed(1)} T
            </button>
          ))}
        </div>
      </div>

      {/* Trajectory & Acceleration */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-[10px] font-semibold uppercase text-slate-400 block mb-1">
            Trayectoria k-Space
          </label>
          <select
            disabled={isTraining}
            value={config.trajectory}
            onChange={(e) =>
              onChangeConfig({ trajectory: e.target.value as TrajectoryType })
            }
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-slate-200"
          >
            <option value="poisson">Poisson-disc 2D</option>
            <option value="radial">Radial Áurea</option>
            <option value="cartesian">Cartesiana VD</option>
          </select>
        </div>

        <div>
          <label className="text-[10px] font-semibold uppercase text-slate-400 block mb-1">
            Factor Submuestreo R
          </label>
          <div className="grid grid-cols-2 gap-1">
            {([4, 8] as AccelerationFactor[]).map((acc) => (
              <button
                key={acc}
                type="button"
                disabled={isTraining}
                onClick={() => onChangeConfig({ acceleration: acc })}
                className={`py-1.5 rounded-lg text-xs font-mono font-semibold border transition ${
                  config.acceleration === acc
                    ? "bg-indigo-500/25 border-indigo-400 text-indigo-200"
                    : "bg-slate-950 border-slate-800 text-slate-400"
                }`}
              >
                R={acc}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Paper 2 Scientific Contributions Toggles */}
      <div className="space-y-2 bg-slate-950/90 border border-slate-800 rounded-xl p-3">
        <div className="text-[11px] font-semibold text-cyan-300 flex items-center gap-1.5">
          <Cpu className="w-3.5 h-3.5" />
          3. Contribuciones Físicas Paper 2
        </div>

        {/* Toggle 1: Exact Convection-Reaction vs Helmholtz */}
        <label className="flex items-start justify-between gap-2 cursor-pointer pt-1">
          <div>
            <span className="text-xs font-medium text-slate-200 block">
              EDP Convección-Reacción (∇σ ≠ 0)
            </span>
            <span className="text-[10px] text-slate-400 block">
              {config.use_convection_reaction
                ? "Activo: cr-PINN evalúa ∇σ·∇φ⁺ vía Autograd"
                : "Inactivo: Helmholtz-PINN (Paper 1, ∇σ ≈ 0)"}
            </span>
          </div>
          <input
            type="checkbox"
            disabled={isTraining}
            checked={config.use_convection_reaction}
            onChange={(e) =>
              onChangeConfig({ use_convection_reaction: e.target.checked })
            }
            className="mt-1 accent-emerald-400 w-4 h-4 cursor-pointer"
          />
        </label>

        {/* Toggle 2: GradNorm Dynamic Weighting */}
        <label className="flex items-start justify-between gap-2 cursor-pointer border-t border-slate-800/80 pt-2">
          <div>
            <span className="text-xs font-medium text-slate-200 block">
              Balanceo Auto-Adaptativo GradNorm
            </span>
            <span className="text-[10px] text-slate-400 block">
              Wang et al. (2021): auto-calibra λ_ept y λ_harm cada 5 épocas
            </span>
          </div>
          <input
            type="checkbox"
            disabled={isTraining}
            checked={config.use_gradnorm}
            onChange={(e) => onChangeConfig({ use_gradnorm: e.target.checked })}
            className="mt-1 accent-cyan-400 w-4 h-4 cursor-pointer"
          />
        </label>
      </div>

      {/* Hyperparameters */}
      <div className="space-y-2">
        <div className="flex justify-between text-[11px]">
          <span className="text-slate-400 flex items-center gap-1">
            <Sliders className="w-3 h-3" /> Épocas SIREN GPU:
          </span>
          <span className="font-mono text-slate-200">{config.epochs}</span>
        </div>
        <input
          type="range"
          min={40}
          max={300}
          step={20}
          disabled={isTraining}
          value={config.epochs}
          onChange={(e) => onChangeConfig({ epochs: Number(e.target.value) })}
          className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
        />
      </div>

      {/* Action Buttons */}
      <div className="pt-1">
        {!isTraining ? (
          <button
            type="button"
            onClick={onStartTraining}
            className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-500 hover:from-cyan-400 hover:to-emerald-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20 transition"
          >
            <Play className="w-4 h-4 fill-current" />
            Ejecutar Inversión Electro-Térmica en GPU
          </button>
        ) : (
          <button
            type="button"
            onClick={onStopTraining}
            className="w-full py-2.5 px-4 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs flex items-center justify-center gap-2 shadow-lg transition"
          >
            <Square className="w-4 h-4 fill-current" />
            Detener Entrenamiento PINN
          </button>
        )}
      </div>
    </aside>
  );
};
