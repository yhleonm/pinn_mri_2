import React, { useEffect, useRef, useState } from "react";
import { Crosshair, Flame, Layers, Thermometer, Zap } from "lucide-react";
import { ElectroThermalMapsPayload } from "../types/mri";

interface ElectroThermalViewerProps {
  maps: ElectroThermalMapsPayload | null;
  selectedRow: number;
  onSelectRow: (row: number) => void;
  showCrosshair: boolean;
}

type ColormapName = "gray" | "turbo" | "inferno" | "thermal" | "viridis";

function sampleColormap(t: number, cmap: ColormapName): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  if (cmap === "gray") {
    const v = Math.round(x * 255);
    return [v, v, v];
  }
  if (cmap === "turbo") {
    // High-contrast scientific colormap for conductivity sigma (S/m)
    const r = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * x - 3))));
    const g = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * x - 2))));
    const b = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * x - 1))));
    return [r, g, b];
  }
  if (cmap === "inferno") {
    // SAR (W/kg)
    const r = Math.round(255 * Math.pow(x, 0.55));
    const g = Math.round(255 * Math.pow(x, 1.65));
    const b = Math.round(255 * (x < 0.5 ? x * 1.4 : Math.max(0, (1 - x) * 1.6)));
    return [r, g, b];
  }
  if (cmap === "thermal") {
    // Absolute Temperature T (°C): deep blue (37°C) -> cyan -> amber -> crimson/white (41°C+)
    if (x < 0.25) {
      const u = x / 0.25;
      return [Math.round(15 * u), Math.round(40 + 160 * u), Math.round(130 + 115 * u)];
    } else if (x < 0.55) {
      const u = (x - 0.25) / 0.3;
      return [Math.round(15 + 230 * u), Math.round(200 + 40 * u), Math.round(245 * (1 - u))];
    } else if (x < 0.85) {
      const u = (x - 0.55) / 0.3;
      return [250, Math.round(240 - 180 * u), Math.round(20 * u)];
    } else {
      const u = (x - 0.85) / 0.15;
      return [255, Math.round(60 + 195 * u), Math.round(20 + 235 * u)];
    }
  }
  // viridis fallback
  const r = Math.round(255 * Math.min(1, Math.max(0, -0.2 + 1.3 * x)));
  const g = Math.round(255 * Math.min(1, Math.max(0, 0.1 + 0.85 * x)));
  const b = Math.round(255 * Math.min(1, Math.max(0, 0.55 + 0.4 * (1 - x))));
  return [r, g, b];
}

interface MatrixCanvasCardProps {
  title: string;
  subtitle: string;
  unit: string;
  matrix: number[][];
  cmap: ColormapName;
  vmin: number;
  vmax: number;
  badgeText?: string;
  badgeColor?: string;
  hoverCoord: { r: number; c: number } | null;
  onHoverCoord: (coord: { r: number; c: number } | null) => void;
  selectedRow: number;
  onSelectRow: (row: number) => void;
  showCrosshair: boolean;
}

const MatrixCanvasCard: React.FC<MatrixCanvasCardProps> = ({
  title,
  subtitle,
  unit,
  matrix,
  cmap,
  vmin,
  vmax,
  badgeText,
  badgeColor = "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
  hoverCoord,
  onHoverCoord,
  selectedRow,
  onSelectRow,
  showCrosshair,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rows = matrix?.length || 128;
  const cols = matrix?.[0]?.length || 128;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !matrix || matrix.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imgData = ctx.createImageData(cols, rows);
    const denom = Math.max(1e-8, vmax - vmin);

    for (let r = 0; r < rows; r++) {
      const rowArr = matrix[r];
      for (let c = 0; c < cols; c++) {
        const val = rowArr[c] ?? 0;
        const norm = (val - vmin) / denom;
        const [R, G, B] = sampleColormap(norm, cmap);
        const idx = (r * cols + c) * 4;
        imgData.data[idx] = R;
        imgData.data[idx + 1] = G;
        imgData.data[idx + 2] = B;
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Draw horizontal 1D profile line (dashed cyan/amber line across ventricles + glioma)
    if (showCrosshair && selectedRow >= 0 && selectedRow < rows) {
      ctx.save();
      ctx.strokeStyle = "rgba(34, 211, 238, 0.85)";
      ctx.lineWidth = 1.0;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(0, selectedRow + 0.5);
      ctx.lineTo(cols, selectedRow + 0.5);
      ctx.stroke();
      ctx.restore();
    }

    if (hoverCoord) {
      ctx.save();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.arc(hoverCoord.c + 0.5, hoverCoord.r + 0.5, 3.0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }, [matrix, cmap, vmin, vmax, rows, cols, selectedRow, hoverCoord, showCrosshair]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const c = Math.min(cols - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * cols)));
    const r = Math.min(rows - 1, Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * rows)));
    onHoverCoord({ r, c });
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const r = Math.min(rows - 1, Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * rows)));
    onSelectRow(r);
  };

  const hoveredValue =
    hoverCoord && matrix?.[hoverCoord.r]?.[hoverCoord.c] !== undefined
      ? matrix[hoverCoord.r][hoverCoord.c]
      : null;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-2.5 flex flex-col justify-between shadow-lg hover:border-slate-700 transition-colors">
      <div className="flex items-start justify-between gap-1 mb-1.5">
        <div>
          <h4 className="text-xs font-semibold text-slate-100 tracking-tight flex items-center gap-1.5">
            {title}
          </h4>
          <p className="text-[10px] text-slate-400">{subtitle}</p>
        </div>
        {badgeText && (
          <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${badgeColor}`}>
            {badgeText}
          </span>
        )}
      </div>

      <div className="relative aspect-square w-full bg-slate-950 rounded-lg overflow-hidden border border-slate-800/80 flex items-center justify-center">
        <canvas
          ref={canvasRef}
          width={cols}
          height={rows}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => onHoverCoord(null)}
          onClick={handleClick}
          className="w-full h-full object-contain cursor-crosshair"
          style={{ imageRendering: "pixelated" }}
        />
        {hoveredValue !== null && (
          <div className="absolute bottom-1.5 left-1.5 bg-slate-950/90 border border-slate-700/80 rounded px-1.5 py-0.5 text-[10px] font-mono text-cyan-200 pointer-events-none">
            ({hoverCoord?.c}, {hoverCoord?.r}): {hoveredValue.toFixed(3)} {unit}
          </div>
        )}
      </div>

      <div className="mt-2 flex items-center justify-between text-[10px] font-mono text-slate-400">
        <span>{vmin.toFixed(2)}</span>
        <span className="text-slate-300">{unit}</span>
        <span>{vmax.toFixed(2)}</span>
      </div>
    </div>
  );
};

export const ElectroThermalViewer: React.FC<ElectroThermalViewerProps> = ({
  maps,
  selectedRow,
  onSelectRow,
  showCrosshair,
}) => {
  const [hoverCoord, setHoverCoord] = useState<{ r: number; c: number } | null>(null);
  const [secondaryMode, setSecondaryMode] = useState<"helmholtz_vs_cr" | "cem43_vs_perfusion">("helmholtz_vs_cr");

  if (!maps) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-8 text-center text-slate-400">
        Cargando tensores electro-térmicos en GPU...
      </div>
    );
  }

  // Dynamic ranges for scientific display
  const tempMatrix = maps.temp_map_c || [];
  let maxTemp = 39.5;
  for (const row of tempMatrix) {
    for (const v of row) {
      if (v > maxTemp) maxTemp = v;
    }
  }
  const vmaxTemp = Math.max(38.5, Math.min(43.5, maxTemp + 0.25));

  let maxSar = 2.5;
  for (const row of maps.sar_map || []) {
    for (const v of row) {
      if (v > maxSar) maxSar = v;
    }
  }

  let maxCem = 0.05;
  for (const row of maps.cem43_map || []) {
    for (const v of row) {
      if (v > maxCem) maxCem = v;
    }
  }

  return (
    <div className="space-y-2.5">
      {/* Header bar for the 5-panel synchronized PACS 2.0 viewer */}
      <div className="flex flex-wrap items-center justify-between gap-2 bg-slate-900/80 border border-slate-800 px-3 py-2 rounded-xl">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold text-slate-100">
            Visor Sincronizado Multi-Física PACS 2.0 (5 Vistas Acopladas)
          </span>
          <span className="text-[11px] font-mono text-cyan-300 bg-cyan-950/60 border border-cyan-800/50 px-2 py-0.5 rounded">
            Corte Transversal 1D: Fila y={selectedRow} (clic en cualquier mapa para mover perfil)
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setSecondaryMode("helmholtz_vs_cr")}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition ${
              secondaryMode === "helmholtz_vs_cr"
                ? "bg-cyan-500/20 text-cyan-200 border border-cyan-500/40"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
          >
            Vista 3: Helmholtz (Paper 1) vs cr-PINN (Paper 2)
          </button>
          <button
            type="button"
            onClick={() => setSecondaryMode("cem43_vs_perfusion")}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition ${
              secondaryMode === "cem43_vs_perfusion"
                ? "bg-amber-500/20 text-amber-200 border border-amber-500/40"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
          >
            Vista 3: Dosis Térmica CEM43 / Perfusión ω_b(x)
          </button>
        </div>
      </div>

      {/* 5 Synchronized Medical Panels */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-2.5">
        {/* Panel 1: Reconstructed Magnitude |m(x)| */}
        <MatrixCanvasCard
          title="1. Magnitud |m(x)|"
          subtitle="SENSE-PINN Libre de Aliasing"
          unit="a.u."
          matrix={maps.mag_pred}
          cmap="gray"
          vmin={0.0}
          vmax={1.0}
          badgeText={`${maps.b0_tesla.toFixed(1)} T`}
          hoverCoord={hoverCoord}
          onHoverCoord={setHoverCoord}
          selectedRow={selectedRow}
          onSelectRow={onSelectRow}
          showCrosshair={showCrosshair}
        />

        {/* Panel 2: Exact Convection-Reaction Conductivity sigma (Paper 2) */}
        <MatrixCanvasCard
          title="2. σ(x) cr-PINN (Paper 2)"
          subtitle="EDP Convección-Reacción (∇σ ≠ 0)"
          unit="S/m"
          matrix={maps.sigma_pred}
          cmap="turbo"
          vmin={0.0}
          vmax={2.0}
          badgeText="cr-EPT Exacto"
          badgeColor="bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
          hoverCoord={hoverCoord}
          onHoverCoord={setHoverCoord}
          selectedRow={selectedRow}
          onSelectRow={onSelectRow}
          showCrosshair={showCrosshair}
        />

        {/* Panel 3: Helmholtz Reference (Paper 1) OR CEM43 / Perfusion */}
        {secondaryMode === "helmholtz_vs_cr" ? (
          <MatrixCanvasCard
            title="3. σ(x) Helmholtz (Paper 1)"
            subtitle="Aprox. Homogénea (∇σ ≈ 0)"
            unit="S/m"
            matrix={maps.sigma_helmholtz_ref || maps.sigma_gt}
            cmap="turbo"
            vmin={0.0}
            vmax={2.0}
            badgeText="-16% Borde/LCR"
            badgeColor="bg-amber-500/20 text-amber-300 border-amber-500/30"
            hoverCoord={hoverCoord}
            onHoverCoord={setHoverCoord}
            selectedRow={selectedRow}
            onSelectRow={onSelectRow}
            showCrosshair={showCrosshair}
          />
        ) : (
          <MatrixCanvasCard
            title="3. Dosis Térmica CEM43"
            subtitle="Minutos Equiv. acumulados a 43 °C"
            unit="min"
            matrix={maps.cem43_map}
            cmap="inferno"
            vmin={0.0}
            vmax={Math.max(0.05, maxCem)}
            badgeText="CEM43"
            badgeColor="bg-rose-500/20 text-rose-300 border-rose-500/30"
            hoverCoord={hoverCoord}
            onHoverCoord={setHoverCoord}
            selectedRow={selectedRow}
            onSelectRow={onSelectRow}
            showCrosshair={showCrosshair}
          />
        )}

        {/* Panel 4: Local SAR (W/kg) */}
        <MatrixCanvasCard
          title="4. Deposición SAR(x)"
          subtitle="Fuente Electromagnética σ|E|²/2ρ"
          unit="W/kg"
          matrix={maps.sar_map}
          cmap="inferno"
          vmin={0.0}
          vmax={Math.max(2.0, maxSar)}
          badgeText="SAR Local"
          badgeColor="bg-orange-500/20 text-orange-300 border-orange-500/30"
          hoverCoord={hoverCoord}
          onHoverCoord={setHoverCoord}
          selectedRow={selectedRow}
          onSelectRow={onSelectRow}
          showCrosshair={showCrosshair}
        />

        {/* Panel 5: Absolute Temperature T(x,t) in °C (Pennes Bioheat) */}
        <MatrixCanvasCard
          title="5. Biocalor Pennes T(x,t)"
          subtitle="Trampa Térmica por Hipoperfusión"
          unit="°C"
          matrix={maps.temp_map_c}
          cmap="thermal"
          vmin={37.0}
          vmax={vmaxTemp}
          badgeText={`Max ${maxTemp.toFixed(2)} °C`}
          badgeColor={
            maxTemp >= 39.0
              ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
              : "bg-cyan-500/20 text-cyan-300 border-cyan-500/30"
          }
          hoverCoord={hoverCoord}
          onHoverCoord={setHoverCoord}
          selectedRow={selectedRow}
          onSelectRow={onSelectRow}
          showCrosshair={showCrosshair}
        />
      </div>
    </div>
  );
};
