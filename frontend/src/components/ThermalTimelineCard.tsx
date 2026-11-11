import React, { useState } from "react";
import { AlertTriangle, CheckCircle2, Clock, Flame, ShieldAlert, Sliders } from "lucide-react";
import { ThermalCurves, ThermalSimulationResponse } from "../types/mri";

interface ThermalTimelineCardProps {
  thermalCurves: ThermalCurves | undefined;
  tMaxC: number;
  deltaTTumorC: number;
  cem43MaxMin: number;
  iecSafetyStatus: string;
  onThermalUpdate: (res: ThermalSimulationResponse) => void;
}

export const ThermalTimelineCard: React.FC<ThermalTimelineCardProps> = ({
  thermalCurves,
  tMaxC,
  deltaTTumorC,
  cem43MaxMin,
  iecSafetyStatus,
  onThermalUpdate,
}) => {
  const [durationMinutes, setDurationMinutes] = useState<number>(30);
  const [hypoperfusionFactor, setHypoperfusionFactor] = useState<number>(0.25);
  const [simulating, setSimulating] = useState<boolean>(false);

  const runSimulation = async (newDuration: number, newHypo: number) => {
    setSimulating(true);
    try {
      const resp = await fetch("/api/thermal-simulation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          duration_min: newDuration,
          tumor_perfusion_ratio: newHypo,
          rf_duty_scale: 5.6,
        }),
      });
      if (resp.ok) {
        const data: ThermalSimulationResponse = await resp.json();
        onThermalUpdate(data);
      }
    } catch (err) {
      console.error("Error running Pennes bioheat simulation:", err);
    } finally {
      setSimulating(false);
    }
  };

  const times = thermalCurves?.time_min || [0, 5, 10, 15, 20, 25, 30];
  const tumorCurve = thermalCurves?.t_tumor_c || [37.0, 38.0, 38.6, 39.0, 39.3, 39.5, 39.6];
  const ventricleCurve = thermalCurves?.t_ventricle_c || [37.0, 37.4, 37.7, 37.9, 38.0, 38.1, 38.1];
  const healthyCurve = thermalCurves?.t_healthy_c || [37.0, 37.2, 37.3, 37.4, 37.4, 37.4, 37.4];

  const svgW = 560;
  const svgH = 200;
  const padL = 42;
  const padR = 20;
  const padT = 16;
  const padB = 26;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;

  const yMin = 37.0;
  const yMax = Math.max(40.8, ...tumorCurve, tMaxC);
  const tMaxTime = Math.max(1, times[times.length - 1] || 30);

  const toX = (t: number) => padL + (t / tMaxTime) * plotW;
  const toY = (temp: number) =>
    padT + plotH - ((Math.min(yMax, Math.max(yMin, temp)) - yMin) / (yMax - yMin)) * plotH;

  const buildCurve = (vals: number[]) =>
    vals
      .map((v, idx) => `${idx === 0 ? "M" : "L"}${toX(times[idx] ?? idx).toFixed(1)},${toY(v).toFixed(1)}`)
      .join(" ");

  const isCritical = tMaxC >= 40.0 || iecSafetyStatus.includes("CRITICAL") || iecSafetyStatus.includes("LEVEL 2");
  const isControlled = !isCritical && (tMaxC >= 38.0 || iecSafetyStatus.includes("LEVEL 1"));

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5 shadow-lg flex flex-col justify-between">
      <div className="flex flex-wrap items-start justify-between gap-2 mb-2">
        <div>
          <h3 className="text-xs font-semibold text-slate-100 flex items-center gap-1.5">
            <Flame className="w-4 h-4 text-rose-400" />
            Dosimetría Biocalórica Transitoria de Pennes T(x,t) y Dosis CEM43
          </h3>
          <p className="text-[11px] text-slate-400">
            Acoplamiento SAR → Biocalor con hipoperfusión intratumoral necrótica (ω_b,tumor ={" "}
            {(hypoperfusionFactor * 100).toFixed(0)}% ω_b,sano)
          </p>
        </div>

        {/* IEC 60601-2-33 Safety Badge */}
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-[11px] font-medium ${
            isCritical
              ? "bg-rose-950/70 border-rose-500/50 text-rose-200"
              : isControlled
                ? "bg-amber-950/70 border-amber-500/50 text-amber-200"
                : "bg-emerald-950/70 border-emerald-500/50 text-emerald-200"
          }`}
        >
          {isCritical ? (
            <ShieldAlert className="w-3.5 h-3.5 text-rose-400 shrink-0" />
          ) : isControlled ? (
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          ) : (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          )}
          <span>{iecSafetyStatus || "IEC 60601-2-33 NORMAL (< 38.0 °C)"}</span>
        </div>
      </div>

      {/* Interactive Sliders for Exposure Time and Hypoperfusion */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-2.5 bg-slate-950/70 border border-slate-800/80 rounded-lg p-2.5">
        <div>
          <div className="flex justify-between text-[11px] mb-1">
            <span className="text-slate-300 flex items-center gap-1">
              <Clock className="w-3 h-3 text-cyan-400" /> Exposición RF Continua:
            </span>
            <span className="font-mono text-cyan-300 font-semibold">{durationMinutes} min</span>
          </div>
          <input
            type="range"
            min={2}
            max={45}
            step={1}
            value={durationMinutes}
            disabled={simulating}
            onChange={(e) => {
              const v = Number(e.target.value);
              setDurationMinutes(v);
              runSimulation(v, hypoperfusionFactor);
            }}
            className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
        </div>

        <div>
          <div className="flex justify-between text-[11px] mb-1">
            <span className="text-slate-300 flex items-center gap-1">
              <Sliders className="w-3 h-3 text-rose-400" /> Perfusión Tumoral (ω_b,tum / ω_b,sano):
            </span>
            <span className="font-mono text-rose-300 font-semibold">
              {(hypoperfusionFactor * 100).toFixed(0)}% ({(9.0 * hypoperfusionFactor).toFixed(2)} kg/m³s)
            </span>
          </div>
          <input
            type="range"
            min={0.05}
            max={1.0}
            step={0.05}
            value={hypoperfusionFactor}
            disabled={simulating}
            onChange={(e) => {
              const v = Number(e.target.value);
              setHypoperfusionFactor(v);
              runSimulation(durationMinutes, v);
            }}
            className="w-full accent-rose-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
        </div>
      </div>

      {/* Transient Heating SVG Plot */}
      <div className="w-full overflow-hidden bg-slate-950/90 rounded-lg border border-slate-800/80 p-1">
        <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full h-44">
          {/* Grid lines and IEC thresholds */}
          {[37.0, 38.0, 39.0, 40.0, 41.0].map((tempVal) => {
            if (tempVal > yMax) return null;
            const y = toY(tempVal);
            const isIecNormal = Math.abs(tempVal - 39.0) < 0.01;
            const isIecCrit = Math.abs(tempVal - 40.0) < 0.01;
            return (
              <g key={tempVal}>
                <line
                  x1={padL}
                  y1={y}
                  x2={svgW - padR}
                  y2={y}
                  stroke={isIecCrit ? "#f43f5e" : isIecNormal ? "#f59e0b" : "#1e293b"}
                  strokeDasharray={isIecCrit || isIecNormal ? "4 2" : "3 3"}
                  strokeWidth={isIecCrit || isIecNormal ? "1.1" : "1"}
                  opacity={isIecCrit || isIecNormal ? "0.7" : "1"}
                />
                <text
                  x={padL - 5}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-slate-400 text-[9px] font-mono"
                >
                  {tempVal.toFixed(1)}°C
                </text>
              </g>
            );
          })}

          {/* IEC Threshold Labels */}
          <text
            x={svgW - padR - 4}
            y={toY(39.0) - 3}
            textAnchor="end"
            className="fill-amber-400/80 text-[8px] font-mono"
          >
            Límite IEC 1er Nivel (39.0 °C)
          </text>
          {yMax >= 40.0 && (
            <text
              x={svgW - padR - 4}
              y={toY(40.0) - 3}
              textAnchor="end"
              className="fill-rose-400/90 text-[8px] font-mono"
            >
              Límite IEC 2do Nivel (40.0 °C)
            </text>
          )}

          {/* Curves */}
          <path d={buildCurve(healthyCurve)} fill="none" stroke="#38bdf8" strokeWidth="1.6" />
          <path
            d={buildCurve(ventricleCurve)}
            fill="none"
            stroke="#a855f7"
            strokeWidth="1.6"
            strokeDasharray="3 2"
          />
          <path d={buildCurve(tumorCurve)} fill="none" stroke="#f43f5e" strokeWidth="2.4" />

          {/* X-axis labels */}
          {times
            .filter((_, i) => i % Math.max(1, Math.floor(times.length / 6)) === 0 || i === times.length - 1)
            .map((tVal) => (
              <text
                key={tVal}
                x={toX(tVal)}
                y={svgH - 6}
                textAnchor="middle"
                className="fill-slate-400 text-[9px] font-mono"
              >
                {tVal.toFixed(0)}m
              </text>
            ))}
        </svg>
      </div>

      {/* Bottom KPI strip */}
      <div className="mt-2 grid grid-cols-3 gap-2 text-[11px]">
        <div className="bg-slate-950/80 border border-slate-800 rounded-lg px-2.5 py-1.5 flex items-center justify-between">
          <span className="text-slate-400">Pico Hotspot T_max:</span>
          <span className="font-mono font-bold text-rose-300">{tMaxC.toFixed(2)} °C</span>
        </div>
        <div className="bg-slate-950/80 border border-slate-800 rounded-lg px-2.5 py-1.5 flex items-center justify-between">
          <span className="text-slate-400">Trampa Térmica ΔT:</span>
          <span className="font-mono font-bold text-amber-300">+{deltaTTumorC.toFixed(2)} °C</span>
        </div>
        <div className="bg-slate-950/80 border border-slate-800 rounded-lg px-2.5 py-1.5 flex items-center justify-between">
          <span className="text-slate-400">Dosis CEM43:</span>
          <span className="font-mono font-bold text-cyan-300">{cem43MaxMin.toFixed(4)} min</span>
        </div>
      </div>
    </div>
  );
};
