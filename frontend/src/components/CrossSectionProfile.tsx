import React from "react";
import { Activity, RotateCcw } from "lucide-react";
import { ElectroThermalMapsPayload } from "../types/mri";

interface CrossSectionProfileProps {
  maps: ElectroThermalMapsPayload | null;
  selectedRow: number;
  onResetRow: () => void;
}

export const CrossSectionProfile: React.FC<CrossSectionProfileProps> = ({
  maps,
  selectedRow,
  onResetRow,
}) => {
  if (!maps || !maps.sigma_pred || maps.sigma_pred.length === 0) {
    return null;
  }

  const H = maps.grid_size || maps.sigma_pred.length;
  const W = maps.sigma_pred[0]?.length || 128;
  const safeRow = Math.min(H - 1, Math.max(0, selectedRow));

  const sigmaGtRow = maps.sigma_gt?.[safeRow] || new Array(W).fill(0);
  const sigmaHelmRow = maps.sigma_helmholtz_ref?.[safeRow] || new Array(W).fill(0);
  const sigmaCrRow = maps.sigma_pred?.[safeRow] || new Array(W).fill(0);
  const tempRow = maps.temp_map_c?.[safeRow] || new Array(W).fill(37.0);

  // SVG dimensions
  const svgW = 680;
  const svgH = 220;
  const padL = 44;
  const padR = 44;
  const padT = 18;
  const padB = 28;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;

  const yMaxSigma = 2.0;
  const yMinTemp = 37.0;
  const yMaxTemp = Math.max(40.5, ...tempRow);

  const toX = (i: number) => padL + (i / Math.max(1, W - 1)) * plotW;
  const toYSigma = (s: number) =>
    padT + plotH - (Math.min(yMaxSigma, Math.max(0, s)) / yMaxSigma) * plotH;
  const toYTemp = (t: number) =>
    padT +
    plotH -
    ((Math.min(yMaxTemp, Math.max(yMinTemp, t)) - yMinTemp) /
      Math.max(0.5, yMaxTemp - yMinTemp)) *
      plotH;

  const buildPath = (arr: number[], mapY: (v: number) => number) =>
    arr
      .map((v, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${mapY(v).toFixed(1)}`)
      .join(" ");

  const pathGt = buildPath(sigmaGtRow, toYSigma);
  const pathHelm = buildPath(sigmaHelmRow, toYSigma);
  const pathCr = buildPath(sigmaCrRow, toYSigma);
  const pathTemp = buildPath(tempRow, toYTemp);

  // Peak metrics on the current line
  const maxGt = Math.max(...sigmaGtRow);
  const maxHelm = Math.max(...sigmaHelmRow);
  const maxCr = Math.max(...sigmaCrRow);
  const maxTempLine = Math.max(...tempRow);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5 shadow-lg flex flex-col justify-between">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div>
          <h3 className="text-xs font-semibold text-slate-100 flex items-center gap-1.5">
            <Activity className="w-4 h-4 text-cyan-400" />
            Perfil Transversal 1D (Ventrículos LCR + Glioma) — Fila y = {safeRow}
          </h3>
          <p className="text-[11px] text-slate-400">
            Demostración de recuperación de bordes exactos (∇σ · ∇φ⁺) frente al truncamiento de Helmholtz (∇σ ≈ 0)
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onResetRow}
            className="flex items-center gap-1 text-[11px] bg-slate-800 hover:bg-slate-700 text-cyan-300 px-2 py-1 rounded border border-slate-700 transition"
          >
            <RotateCcw className="w-3 h-3" />
            Fila 57 (LCR + Glioma)
          </button>
        </div>
      </div>

      {/* Legend + Quick Peak Readouts */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1.5 text-[11px]">
        <div className="flex flex-wrap items-center gap-3">
          <span className="flex items-center gap-1.5 text-slate-300">
            <span className="w-3 h-0.5 bg-slate-400 inline-block border-b border-dashed border-slate-300" />
            Ground Truth (Max {maxGt.toFixed(2)} S/m)
          </span>
          <span className="flex items-center gap-1.5 text-amber-300">
            <span className="w-3 h-0.5 bg-amber-400 inline-block" />
            Helmholtz-PINN Paper 1 (Max {maxHelm.toFixed(2)} S/m)
          </span>
          <span className="flex items-center gap-1.5 text-emerald-300 font-semibold">
            <span className="w-3 h-0.5 bg-emerald-400 inline-block" />
            cr-PINN Paper 2 (Max {maxCr.toFixed(2)} S/m)
          </span>
          <span className="flex items-center gap-1.5 text-rose-300">
            <span className="w-3 h-0.5 bg-rose-400 inline-block" />
            Temp. Pennes T(x) (Max {maxTempLine.toFixed(2)} °C)
          </span>
        </div>
      </div>

      {/* SVG Chart */}
      <div className="w-full overflow-hidden bg-slate-950/90 rounded-lg border border-slate-800/80 p-1">
        <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full h-48">
          {/* Horizontal grid lines */}
          {[0.0, 0.5, 1.0, 1.5, 2.0].map((val) => {
            const y = toYSigma(val);
            return (
              <g key={val}>
                <line
                  x1={padL}
                  y1={y}
                  x2={svgW - padR}
                  y2={y}
                  stroke="#1e293b"
                  strokeDasharray="3 3"
                  strokeWidth="1"
                />
                <text
                  x={padL - 6}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-slate-400 text-[9px] font-mono"
                >
                  {val.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Right Axis Labels for Temperature (°C) */}
          {[37.0, 38.5, 40.0].map((tVal) => {
            const y = toYTemp(tVal);
            return (
              <text
                key={tVal}
                x={svgW - padR + 6}
                y={y + 3}
                textAnchor="start"
                className="fill-rose-400/80 text-[9px] font-mono"
              >
                {tVal.toFixed(1)}°C
              </text>
            );
          })}

          {/* Anatomical region annotations for Row 57 */}
          {Math.abs(safeRow - Math.round(0.445 * H)) <= 6 && (
            <>
              <rect
                x={toX(Math.round(0.39 * W))}
                y={padT}
                width={toX(Math.round(0.61 * W)) - toX(Math.round(0.39 * W))}
                height={plotH}
                fill="rgba(56, 189, 248, 0.05)"
              />
              <text
                x={toX(Math.round(0.5 * W))}
                y={padT + 11}
                textAnchor="middle"
                className="fill-cyan-400/80 text-[9px] font-mono"
              >
                Ventrículos LCR (1.65 S/m)
              </text>

              <rect
                x={toX(Math.round(0.62 * W))}
                y={padT}
                width={toX(Math.round(0.75 * W)) - toX(Math.round(0.62 * W))}
                height={plotH}
                fill="rgba(244, 63, 94, 0.06)"
              />
              <text
                x={toX(Math.round(0.685 * W))}
                y={padT + 11}
                textAnchor="middle"
                className="fill-rose-400/90 text-[9px] font-mono"
              >
                Glioma (1.22 S/m)
              </text>
            </>
          )}

          {/* Ground Truth Conductivity (Dashed Slate/White) */}
          {maps.has_ground_truth !== false && (
            <path
              d={pathGt}
              fill="none"
              stroke="#94a3b8"
              strokeWidth="1.6"
              strokeDasharray="4 3"
            />
          )}

          {/* Helmholtz-PINN Paper 1 (Amber) */}
          <path d={pathHelm} fill="none" stroke="#f59e0b" strokeWidth="1.8" opacity="0.9" />

          {/* Temperature Profile T(x) (Rose translucent) */}
          <path
            d={pathTemp}
            fill="none"
            stroke="#fb7185"
            strokeWidth="1.4"
            strokeDasharray="2 2"
            opacity="0.85"
          />

          {/* cr-PINN Paper 2 (Bright Emerald) */}
          <path d={pathCr} fill="none" stroke="#10b981" strokeWidth="2.3" />

          {/* Axis titles */}
          <text
            x={padL}
            y={padT - 5}
            className="fill-emerald-400 text-[9px] font-mono font-semibold"
          >
            σ(x) [S/m]
          </text>
          <text
            x={svgW - padR}
            y={padT - 5}
            textAnchor="end"
            className="fill-rose-400 text-[9px] font-mono font-semibold"
          >
            T(x) [°C]
          </text>
          <text
            x={padL + plotW / 2}
            y={svgH - 6}
            textAnchor="middle"
            className="fill-slate-400 text-[9px] font-mono"
          >
            Coordenada Espacial Transversal x (índice de vóxel 0 → {W - 1})
          </text>
        </svg>
      </div>
    </div>
  );
};
