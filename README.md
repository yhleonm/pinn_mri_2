# `pinn_mri_2` — Multi-Physics Coupled Electro-Thermal PINN (`ElectroThermalMRIPINN`)

**Segunda Generación de Reconstrucción Cuantitativa de Resonancia Magnética y Dosimetría Termodinámica:**
*Tomografía de Propiedades Eléctricas por Convección-Reacción Diferenciable Exacta ($\nabla\sigma \neq 0, \nabla\varepsilon_r \neq 0$), Ecuación Parabólica de Biocalor de Pennes ($T(\mathbf{x}, t)$ en $^\circ\text{C}$ y Dosis $\text{CEM}_{43}$) y Balanceo Auto-Adaptativo por Norma de Gradientes (GradNorm).*

---

## 1. Contribuciones Científicas Originales (Superación de las Limitaciones del Artículo 1)

En el proyecto antecedente (`pinn_mri`), la reconstrucción de conductividad eléctrica ($\sigma$) y permitividad relativa ($\varepsilon_r$) se formuló bajo la aproximación localmente homogénea de Helmholtz ($\nabla\sigma \approx 0, \nabla\varepsilon_r \approx 0$), estimando únicamente el índice cuasi-estático de tasa de absorción específica ($\text{SAR}$) con hiperparámetros de pérdida $\lambda$ fijos.

`pinn_mri_2` resuelve directamente las **tres limitaciones físicas fundamentales** reportadas en la discusión del primer manuscrito:

1. **EDP Completa de Convección-Reacción Libre de Difusión Artificial (`DifferentiableCrEPT` en [`src/physics/cr_ept_operator.py`](file:///C:/Users/yelem/workspace/pinn_mri_2/src/physics/cr_ept_operator.py)):**
   - Elimina el truncamiento de borde $\nabla\sigma \approx 0$ resolviendo la ecuación diferencial parcial completa de fase transceptora:
     $$\nabla_m^2 \phi^+(\mathbf{x}) + \frac{2 \nabla_m |B_1^+| \cdot \nabla_m \phi^+}{|B_1^+| + \epsilon} - \frac{\nabla_m \sigma(\mathbf{x}) \cdot \nabla_m \phi^+(\mathbf{x})}{\sigma(\mathbf{x}) + \epsilon_{\sigma}} = \mu_0 \omega \sigma(\mathbf{x})$$
   - Evalúa analíticamente $\nabla_m \sigma(\mathbf{x})$ y $\nabla_m \phi^+(\mathbf{x})$ sobre representaciones neuronales implícitas periódicas SIREN ($\mathcal{C}^\infty$) mediante diferenciación automática exacta (`torch.autograd.grad(..., create_graph=True)`) con escalado métrico $(2/L_x)$, sin introducir la viscosidad artificial estabilizadora ($\nu_{\text{art}}\nabla^2\sigma$) que degrada al método discreto `cr-MREPT`.
   - **Resultado:** Elimina la subestimación del $-15\%\text{ al }-18\%$ en el glioma ($\hat{\sigma}_{\text{tumor}} \approx 1.19\text{ S/m}$ frente a $\sigma_{\text{GT}} = 1.22\text{ S/m}$) y recupera el valor dieléctrico real del líquido cefalorraquídeo en los ventrículos laterales ($\hat{\sigma}_{\text{LCR}} \approx 1.62\text{ S/m}$ frente a $0.90\text{ S/m}$ en Helmholtz-PINN).

2. **Acoplamiento Electro-Térmico con la Ecuación de Biocalor de Pennes y Dosis $\text{CEM}_{43}$ (`PennesBioheatSolver` en [`src/physics/pennes_bioheat.py`](file:///C:/Users/yelem/workspace/pinn_mri_2/src/physics/pennes_bioheat.py)):**
   - Acopla la fuente electromagnética reconstruida $\text{SAR}(\mathbf{x}) = \frac{\sigma(\mathbf{x})\|\mathbf{E}(\mathbf{x})\|_2^2}{2\rho(\mathbf{x})}$ con la EDP parabólica de transferencia de biocalor de Pennes en régimen estacionario y transitorio $T(\mathbf{x}, t)$ ($t \in [0, 30]\text{ min}$):
     $$\rho(\mathbf{x}) c_p(\mathbf{x}) \frac{\partial T(\mathbf{x}, t)}{\partial t} = \nabla_m \cdot \left( k_t(\mathbf{x}) \nabla_m T(\mathbf{x}, t) \right) - \omega_b(\mathbf{x}) c_b \left( T(\mathbf{x}, t) - T_{\text{art}} \right) + \rho(\mathbf{x}) \text{SAR}(\mathbf{x}) + Q_m(\mathbf{x})$$
   - Modela la fisiología oncológica real con hipoperfusión en el núcleo necrótico/hipóxico tumoral ($\omega_{b,\text{tumor}} = 0.25\,\omega_{b,\text{sano}}$), prediciendo trampas térmicas intratumorales ($\Delta T$ en $^\circ\text{C}$), dosis térmica acumulada de Sapareto-Dewey ($\text{CEM}_{43}$ en minutos equivalentes a $43^\circ\text{C}$) y alertas regulatorias **IEC 60601-2-33**.

3. **Balanceo Auto-Adaptativo de Pérdidas por Norma de Gradientes (`GradNormWeighting` en [`src/training/adaptive_losses.py`](file:///C:/Users/yelem/workspace/pinn_mri_2/src/training/adaptive_losses.py)):**
   - Implementa el esquema de ponderación dinámica de Wang, Teng y Perdikaris (2021), auto-calibrando cada 5 épocas los pesos $\lambda_{\text{ept}}(t)$ y $\lambda_{\text{harm}}(t)$ con suavizado exponencial ($\alpha = 0.85$).

---

## 2. Arquitectura del Repositorio

```text
C:/Users/yelem/workspace/pinn_mri_2/
├── configs/
│   └── electro_thermal_config.yaml      # Configuración multi-física (1.5T, 3.0T, 7.0T, Pennes, GradNorm)
├── data/                                # Cohortes .h5 migradas + datos reales 8 bobinas UC Berkeley / BART
├── src/
│   ├── data/
│   │   ├── kspace_loader.py             # Fantoma dispersivo Cole-Cole + mapas de perfusión ω_b(x) + UC Berkeley
│   │   └── masks.py                     # Máscaras Cartesiana VD, Radial Áurea y Poisson-disc (R=4x, 8x)
│   ├── physics/
│   │   ├── sense_operator.py            # Operador DifferentiableSENSE + SVD + Virtual Body Coil (VBC)
│   │   ├── cr_ept_operator.py           # EDP Convección-Reacción exacta con ∇σ · ∇φ+ vía Autograd
│   │   └── pennes_bioheat.py            # Solver Diferenciable de Biocalor de Pennes T(x,t) [°C] y CEM43
│   ├── models/
│   │   ├── siren.py                     # Capas periódicas SIREN con regla de la cadena métrica (2/L_x)
│   │   └── electro_thermal_pinn.py      # Arquitectura acoplada de 4 cabezas físicas
│   └── training/
│       ├── adaptive_losses.py           # Pérdida Sobolev H^2 + Residuo cr-EPT + GradNorm auto-adaptativo
│       └── trainer.py                   # Loop en GPU RTX 4060 Ti con telemetría electro-térmica en vivo
├── backend/                             # FastAPI + WebSockets (/ws/train y /api/thermal-simulation)
├── frontend/                            # Estación PACS Electro-Térmica 2.0 (React + Vite + TypeScript + Tailwind)
├── paper2_manuscript/                   # Manuscrito LaTeX completo + Figuras a 300 DPI + references.bib
└── run_paper2_benchmark.py              # Suite maestra de ablación, benchmarks N=5 y generación de figuras
```

---

## 3. Ejecución Rápida en GPU (NVIDIA RTX 4060 Ti)

### A. Lanzar la Estación PACS Electro-Térmica 2.0 (Backend + Frontend)
```powershell
./start_workstation.ps1
```
- **API Backend & WebSockets GPU:** `http://localhost:8002` (`http://localhost:8002/docs`)
- **Estación PACS 2.0 (React/Vite):** `http://localhost:5174`

### B. Reproducir el Benchmark Multi-Campo ($N=5$ semillas) y Figuras a 300 DPI
```powershell
python run_paper2_benchmark.py
```
Los resultados estadísticos se exportan automáticamente a `results/paper2_artifacts/paper2_benchmark_metrics.json` y las 5 figuras de publicación a `paper2_manuscript/figures/`.
