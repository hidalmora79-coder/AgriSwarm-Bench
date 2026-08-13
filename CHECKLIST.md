---
titulo: "AgriSwarm-Py — Multi-UAV Swarm Intelligence"
tipo: "proyecto"
estado: "activo"
prioridad: "alta"
fecha_creacion: 2026-06-01
fecha_limite: 2026-09-30
proyecto_relacionado: "Tesis"
---
# Proyecto Multi-UAV Swarm Intelligence — Checklist

> Archivo maestro de estado. Alineado con `Cambios_para_publicar_CEA.docx`.
> Se actualiza al final de cada sesión. Léeme al inicio de cada sesión.

---

## FASE 0 — Infraestructura y Setup ✅

- [x] Entorno Python 3.14+ con `uv` (`venv_maestro`)
- [x] `requirements.txt` con numpy, scipy, pandas, matplotlib, scikit-learn, pytest, python-docx
- [x] `pyproject.toml` con configuración pytest (raíz del workspace, incluye `01_PROYECTO/AgriSwarm-Py/tests`)
- [x] `config/algorithms.yaml` — hiperparámetros PSO, ACO, HYBRID, experimento
- [x] `src/config.py` — loader YAML + CLI overrides (`--param hybrid.rho=0.05`)
- [x] `scripts/validate_fase1.py` — validación integral Fase 1-4 (22 checks)
- [x] `AGENTS.md` — guía de contexto para el asistente
- [x] `CHECKLIST.md` — este archivo

---

## FASE 1 — Datos Reales (Sentinel-2) @CEA

> **Crítico**: todo lo demás se construye sobre esto.

### Descarga y procesamiento ✅
- [x] Escena Sentinel-2 descargada: S2B_20251029 T13RDM, Copernicus Browser
- [x] Bandas B04 (rojo) y B08 (NIR) extraídas del SAFE
- [x] NDVI calculado: `(nir - red) / (nir + red + 1e-10)`, rango [-0.387, 0.704]
- [x] `scripts/process_sentinel2.py` — pipeline de extracción

### Parches 100×100 ✅
- [x] 24 parches estratificados (6 crítico, 6 alto, 6 moderado, 6 saludable)
- [x] `data/patches/` — 29 archivos .npy (24 estratificados + 5 sin estrato `patch_*`)
- [x] `data/patches/patches_metadata.csv`

### Validación ✅
- [x] `scripts/validate_fase1.py` creado: verifica integridad de parches, metadatos, 29×.npy

### Lo que cambia en el artículo ✅
- [x] NDVI real en lugar de sintético — elimina la objeción más frecuente en CEA

---

## FASE 2 — Rediseño Experimental @CEA

### 2A — Escenario sintético (baseline reproducible) ✅
- [x] Grid 100×100 con 3 zonas de estrés controladas vía Gaussianas
- [x] Experimentos: 30 semillas × 3 configs × 300 iteraciones
- [x] `data/raw/synthetic/` — 90 CSVs (3 configs × 30 seeds)
- [x] ⚠️ Originalmente solo 10 semillas (30 CSVs). **Corregido**: suplementadas seeds 10-29 → 90 CSVs. `supplement_seeds.py` eliminado (temp).

### 2B — 5 parches Sentinel-2 reales ✅
- [x] Corridas Monte Carlo completadas sobre escenarios reales
- [x] `data/raw/` — 29 subdirectorios con 30 CSVs cada uno
- [x] Total: 900 corridas, ~18 MB, ~33 min

### 2C — Escalabilidad de flota (N ∈ {5, 10, 20, 30}) ✅
- [x] Correr HYBRID con flota de 5, 10, 20, 30 agentes sobre el parche moderado_13 (15 seeds × 4 N = 60 corridas)
- [x] Reportar TCR vs N (figura de escalabilidad: `data/scalability/scalability_tcr_vs_n.png`)
- [x] Responde a Limitación 2 de la Sección 5.4 con datos empíricos
- Resultados: N=5 → 0.013, N=10 → 0.028, N=20 → 0.049, N=30 → 0.060 (rendimientos decrecientes)
- Script: `run_scalability.py`

### 2D — Sensibilidad de hiperparámetros ✅
- [x] Diseño factorial 3×3: ρ ∈ {0.05, 0.10, 0.20} × β ∈ {1.0, 2.0, 3.0}
- [x] Escenario sintético, HYBRID, 10 semillas × 300 iteraciones (90 corridas)
- [x] Demostrar robustez del framework: variación total TCR = 0.043 (13% de la media)
- [x] Hallazgo: ρ=0.05 óptimo, β tiene efecto secundario. Config default es near-optimal.
- Script: `run_sensitivity.py` | Datos: `data/sensitivity/sensitivity_results.csv`

---

## FASE 3 — Flota Heterogénea (Framework completo) ✅

- [x] `src/agent.py` — Drone con roles: monitor (v=4.0, 120Wh) y sprayer (v=2.5, 180Wh)
- [x] `residual_capacity_ratio` — feromona ponderada por batería restante
- [x] `config/algorithms.yaml` — `fleet_mix: {monitor: 6, sprayer: 4}`
- [x] HybridPSOACO adaptado: velocidad por rol, feromona residual
- [x] PurePSO, PureACO compatibles con drones heterogéneos
- [x] 900 corridas con flota heterogénea completadas
- [x] 4 figuras publicables generadas

---

## FASE 4 — Métricas Agrícolas @CEA

> Post-procesamiento de datos existentes. No requiere nuevos experimentos.

### 4A — Área monitoreada efectiva (ha/misión) ✅
- [x] Si celda = 10m×10m → grid 100×100 = 100 ha
- [x] `area_efectiva = TCR * area_estresada_total_ha`
- [x] HYBRID: 3.75 ha, PSO: 3.41 ha, ACO: 0.71 ha (sobre 12.15 ha estresadas)
- [x] Figura: `data/agricultural_metrics/effective_area.png`

### 4B — Cobertura por unidad de energía (ha/Wh) ✅
- [x] `eficiencia = area_efectiva / (energia_inicial - energia_residual)`
- [x] PSO: 0.0235 ha/Wh (más eficiente por conservar energía), HYBRID: 0.0169 ha/Wh, ACO: 0.0032 ha/Wh

### 4C — Cobertura por zona de estrés ✅
- [x] Tracking de cobertura por zona (3 zonas: core 20×20, sparse 25×25, strip 15×20)
- [x] Figura: `data/agricultural_metrics/zone_coverage_progression.png`
- [x] Cobertura final: 4.2% (Z1), 4.3% (Z2), 2.7% (Z3) — cobertura lenta por dispersión de agentes

### 4D — Cobertura parcial por umbral de tiempo ✅
- [x] TCR en iteraciones 50, 100, 150, 200, 300
- [x] HYBRID converge en iter 50 (TCR=0.308) y se mantiene estable
- [x] Figura: `data/agricultural_metrics/partial_tcr_progression.png`

### Validación transversal ✅
- [x] `scripts/validate_fase1.py` creado y operativo — 22/22 checks PASS
- [x] `data/figures/intermediate/` — 88 figuras por-parche archivadas (regenerables)

---

## FASE 5 — Análisis Comparativo Justo @CEA ✅

> Baselines más competitivos. Punto crítico para revisores CEA.

### 5A — Boustrophedon determinista ✅
- [x] `BoustrophedonCoverage` clase implementada en `src/algorithms.py`
- [x] Método real usado por drones agrícolas comerciales (sweep + descend)
- [x] Integrado en `run_experiments.py` como `BOUSTROPHEDON`
- [x] Ejecutado: 30 seeds × 300 iter = 30 corridas (27s)
- [x] **Resultado**: TCR=1.0000±0.0000 (cobertura total), Energía residual=25.5 Wh
- [x] **Interpretación**: Barrido exhaustivo logra TCR perfecto en zonas contiguas, pero consume 79% de batería

### 5B — PSO multi-niche ✅
- [x] `PSONiche` clase implementada en `src/algorithms.py`
- [x] PSO con partículas asignadas por nicho (misma asignación que HYBRID), sin ACO
- [x] Aísla la contribución del ACO táctico
- [x] Integrado en `run_experiments.py` como `PSO_MULTI_NICHE`
- [x] Ejecutado: 30 seeds × 300 iter = 30 corridas (37s)
- [x] **Resultado**: TCR=0.472±0.066, Energía residual=23.6 Wh
- [x] **Interpretación**: El direccionamiento por nichos NICHE supera a PSO global (TCR +53%), pero el ACO local no añade beneficio adicional en escenarios homogéneos

### Resultados comparativos (n=30, synthetic, iter 299)
| Config | TCR | TCR_std | Energy | Energy_std | MW_p (vs HYBRID) |
|--------|-----|---------|--------|------------|------------------|
| BOUSTROPHEDON | 1.0000 | 0.0000 | 25.5 | 0.1 | <0.0001 |
| PSO_MULTI_NICHE | 0.4717 | 0.0664 | 23.6 | 14.9 | <0.0001 |
| PSO | 0.3081 | 0.0937 | 72.8 | 17.5 | 0.8360 |
| HYBRID | 0.3037 | 0.0451 | 0.0 | 0.0 | — |
| ACO | 0.0634 | 0.0304 | 0.0 | 0.0 | <0.0001 |

### Interpretación para el manuscrito
- Boustrophedon es el baseline más fuerte pero energéticamente costoso
- PSO_MULTI_NICHE demuestra que el direccionamiento por nichos es la contribución clave
- HYBRID no mejora TCR vs PSO pero añade robustez vía feromonas (por confirmar en flotas heterogéneas)
- Figuras: `data/baseline_comparison/baseline_convergence.png`, `baseline_boxplots.png`, `baseline_tradeoff.png`

---

## FASE 6 — Reescritura Orientada a CEA @CEA ✅

> Cambios estructurales en el texto del manuscrito.

### 6A — Introducción ✅
- [x] Primer párrafo reescrito: empieza con problema agrícola (déficit hídrico en Chihuahua, maíz/nogal/chile)
- [x] Cuantificadas pérdidas: >2,500 MDP anuales en tres cultivos (datos SIAP/SAGARPA)
- [x] Estadísticas del abstract y highlights actualizadas (ANOVA, TCR=0.304, sigma≈0)

### 6B — Sección 2.5: Contexto agronómico del NDVI ✅
- [x] Nueva subsección \ref{hybrid:subsec:ndvi_agronomy} en `02_related_work.tex`
- [x] Justificado NDVI < 0.4 con fisiología vegetal: punto de marchitez permanente (-1.5 MPa)
- [x] Conectado con etapa reproductiva R1-R5 en maíz y desarrollo de nuez en nogal
- [x] 14 nuevas referencias agronómicas añadidas a `referencias.bib`

### 6C — Sección 5.5: Implicaciones operacionales ✅
- [x] Nueva subsección \ref{hybrid:subsec:operational} en `05_discusion.tex`
- [x] Cálculo: 8 misiones para cubrir 30 ha estresadas en huerto de 100 ha
- [x] Costo estimado: 4,800 MNP (UAV) vs 3,600-6,750 MNP (inspección manual)
- [x] Ventana temporal: 4 h UAV vs 8-15 h terrestre (50-73% de reducción)
- [x] Conexión con prácticas agronómicas en Chihuahua (85,000 ha de nogal)

### 6D — Terminología ✅
- [x] "mining mode" → "exploitation mode" en `01_introduccion.tex` y `03_metodologia.tex`

### Archivos modificados
| Archivo | Cambio |
|---------|--------|
| `main.tex` | Abstract + highlights corregidos (ANOVA, TCR=0.304, sigma≈0) |
| `contenido/01_introduccion.tex` | Primer párrafo reescrito (problema agrícola + pérdidas); contribution items actualizados |
| `contenido/02_related_work.tex` | Nueva subsección 2.5 (contexto agronómico NDVI) |
| `contenido/05_discusion.tex` | Nueva subsección 5.5 (implicaciones operacionales) |
| `contenido/03_metodologia.tex` | "mining mode" → "exploitation mode" |
| `referencias.bib` | +14 entradas agronómicas (SIAP, Chaves, Jones, Gitelson, etc.) |
| **Compilación** | ✅ 31 pp, xelatex, 0 errores BibTeX, 0 undefined citations |

---

## FASE 7 — Pipeline Científico (Herramienta interna) ✅

> Pipeline integrado con experimentos de AgriSwarm-Py. Validación de datos y manuscrito.

### Capas implementadas

| Capa | Archivo | Función |
|------|---------|---------|
| **ingest** | `src/layers/layer1_ingest.py` | Valida existencia de datos experimentales: 1020 CSVs, 29 parches NDVI, MC resumen, métricas agrícolas, baselines, escalabilidad, sensibilidad |
| **math** | `src/layers/layer2_math.py` | Recalcula TCR por algoritmo desde CSVs, verifica vs. `verdad.yaml` (tol. 5% rel), ejecuta SymPy validator, opcionalmente `validate_fase1.py` |
| **editorial** | `src/layers/layer3_editorial.py` | Verifica archivos LaTeX (main + 6 secciones + BibTeX con 72 entradas), figuras referenciadas, claves de citación |
| **seal** | `src/layers/layer4_seal.py` | Genera `reports/manifest_seal.json` con SHA256 de uv.lock, audit-log, MC CSV, main.tex y conteo de parches |

### Archivos modificados/creados

| Archivo | Cambio |
|---------|--------|
| `src/layers/layer1_ingest.py` | Reescribe: valida datos de AgriSwarm-Py en vez de `data/raw/` vacío |
| `src/layers/layer2_math.py` | Reescribe: carga `results_monte_carlo.csv`, calcula TCR, verifica contra `verdad.yaml` |
| `src/layers/layer3_editorial.py` | Reescribe: verifica LaTeX + figuras + BibTeX |
| `src/layers/layer4_seal.py` | Extiende: incluye experiment hashes + conteo de parches |
| `src/core/sympy_validator.py` | Mejora: mejor manejo de errores, ecuaciones no numéricas no bloquean |
| `verdad.yaml` | Actualiza: constantes experimentales + TCR esperados |
| `main_expanded.tex` | Actualiza: ecuaciones del manuscrito con constantes |
| `run.ps1` | Arregla: ruta a `uv.exe` + DVC via `python -m dvc` |
| `pyproject.toml` | Actualiza: `[dependency-groups]` moderna, sin dev-dependencies deprecada |

### Uso

```powershell
.\run.ps1 -Target all           # Ejecuta 4 capas + SymPy + DVC status
.\run.ps1 -Target ingest        # Solo una capa
.\run.ps1 -Clean                # Limpia caches y artefactos
```

### Validación sintética

| Algoritmo | Manuscrito | Dato de pipeline | ¿Coincide? |
|-----------|-----------|-----------------|------------|
| HYBRID | 0.304 ± 0.044 | 0.304 ± 0.045 | ✅ |
| PSO | 0.308 ± 0.092 | 0.308 ± 0.094 | ✅ |
| ACO | 0.063 ± 0.030 | 0.063 ± 0.030 | ✅ |
| BOUSTROPHEDON | 1.000 ± 0.000 | 1.000 ± 0.000 | ✅ |
| PSO_MULTI_NICHE | 0.472 ± 0.066 | 0.472 ± 0.066 | ✅ |

Pipeline compara contra CSVs individuales en `data/raw/synthetic/` (150 archivos, 30 seeds × 5 configs).

---

## FASE 8 — Referencias y Posicionamiento CEA @CEA ✅

> 5 papers de Computación y Sistemas (2025–2026) añadidos al manuscrito.

### Papers seleccionados

| Ref key | Título | Relevance |
|---------|--------|-----------|
| `Slimani2025` | UAV-based Systems for Advanced Crop Growth Monitoring with Deep Learning Framework in Complex Agriculture | UAV + agricultura de precisión — citado en introducción |
| `AndradeMogollon2025` | Systematic Literature Review of Generative AI and IoT as Key Technologies for Precision Agriculture | Precision agriculture + AI/IoT — citado en introducción |
| `Guajardo2026` | A Novel Cooperative Hybrid Metaheuristic Optimization Method Based on Collective Intelligence | Metaheurística híbrida cooperativa — citado en related work (Sec 2.4) |
| `RomeroBautista2026` | Depth, Spatial, and Temporal Features for Visual Odometry in Unstructured Agricultural Environments | Entornos agrícolas no estructurados — citado en related work (Sec 2.2) |
| `LagunaSanchez2025` | Comparative study of PSO and Differential Evolution Algorithms on a GPU | PSO optimization — citado en related work (Sec 2.4) |

### Archivos modificados
| Archivo | Cambio |
|---------|--------|
| `referencias.bib` | +5 entradas CEA |
| `01_introduccion.tex` | Citados `Slimani2025`, `AndradeMogollon2025` |
| `02_related_work.tex` | Citados `RomeroBautista2026`, `Guajardo2026`, `LagunaSanchez2025` |
| **Compilación** | ✅ 49 entradas, 0 errores, PDF 1.55 MB |

---

## FASE 9 — Carta de Presentación Estratégica @CEA ✅

> Carta de presentación redactada y compilada con xelatex (2 pp).

| Elemento | Estado |
|----------|--------|
| Relevance CS (no robótica) | ✅ Algoritmos PSO-ACO, estabilidad Lyapunov, validación ANOVA |
| NDVI real (Sentinel-2 10 m) | ✅ Destacado en bullets de contribuciones |
| Código en GitHub | ✅ Enlace a AgriSwarm-Bench |
| Implicaciones Chihuahua | ✅ Semiárido, productores, monitoreo eficiente |
| 4 revisores sugeridos | ✅ Altamirano, Valdez, Slimani, López-Pimentel |
| Archivo | `cover_letter.tex` + `cover_letter.pdf` |

---

## Resumen de Archivos Clave

| Archivo | Propósito |
|---------|-----------|
| `config/algorithms.yaml` | Hiperparámetros centralizados |
| `src/agent.py` | Drone con roles heterogéneos |
| `src/algorithms.py` | PSO, ACO, HybridPSOACO |
| `src/environment.py` | Campo NDVI sintético o real (Sentinel-2) |
| `src/metrics.py` | Cobertura, TCR, energía |
| `src/config.py` | Carga YAML + CLI |
| `run_experiments.py` | Orquestador Monte Carlo |
| `scripts/figures_publication.py` | 4 figuras publicables |
| `scripts/validate_fase1.py` | Validación integral |
| `scripts/process_sentinel2.py` | Extracción Sentinel-2 → parches NDVI |
| `data/raw/` | 900 CSVs de resultados |
| `data/figures/intermediate/` | 88 figuras por-parche (archivadas, regenerables) |
| `data/agricultural_metrics/*.png` | Figuras publicables Fase 4 |
| `data/patches/` | 29 parches NDVI reales |
| `resultados_pso_aco_fase1.csv` | Legacy Fase 1 (para tesis) |
| `trayectorias_pso_aco.png` | Trayectorias flota heterogénea |

---

## Estado General

| Fase | Estado | Prioridad |
|------|--------|-----------|
| 0 — Infraestructura | ✅ Completa | — |
| 1 — Datos reales Sentinel-2 | ✅ Completa | Crítica |
| 2A — Escenario sintético | ✅ Completa | Alta |
| 2B — Parches reales (900 runs) | ✅ Completa | Alta |
| 2C — Escalabilidad de flota | ✅ Completa (60 runs, N∈{5,10,20,30}) | Alta |
| 2D — Sensibilidad hiperparámetros | ✅ Completa (90 runs, ρ×β 3×3) | Alta |
| 3 — Flota heterogénea | ✅ Completa | Alta |
| 4 — Métricas agrícolas | ✅ Completa | Alta |
| 5 — Baselines competitivos | ✅ Completa | Alta |
| 6 — Reescritura CEA | ✅ Completa | Alta |
| 7 — Pipeline científico | ✅ Completa | Baja |
| 8 — Referencias CEA | ✅ Completa | Media |
| 9 — Carta de presentación | ✅ Completa | Media |

---

## Última Sesión — Revisión Final Integral

**Fecha**: 2026-06-06 (Sesión 3)
**Logro**: Revisión exhaustiva de fases 0–9 + Fase 7 completada + reconciliación TCR.

### Resultados de verificación

| Ítem | Estado | Detalle |
|------|--------|---------|
| `validate_fase1.py` | ✅ 22/22 PASS | Parches, sintético, reales, escalabilidad, sensibilidad, métricas agrícolas, baselines, scripts |
| Pipeline científico (`run.ps1 -Target all`) | ✅ 4/4 OK + SymPy + DVC | Ingest, Math, Editorial, Seal — todos PASS |
| Compilación manuscrito | ✅ 32 pp, 1.55 MB, 0 errores | Solo warnings cosméticos (acro labels, unicode en PDF strings) |
| Compilación cover letter | ✅ 2 pp, 114 KB | Carta de presentación OK |
| Consistencia TCR (synthetic) | ✅ Todos coinciden | HYBRID=0.304, PSO=0.308, ACO=0.063 — pipeline confirma vs CSVs |
| Cubierta de archivos | ✅ 12/12 scripts, 29/29 parches, 1020/1020 CSVs | Todo presente |

### Hallazgos corregidos durante la revisión

1. **Fase 7 pipeline**: Comparaba contra `results_monte_carlo.csv` (datos agregados de 29 parches reales) en vez de `data/raw/synthetic/` (escenario sintético). Corregido: ahora lee CSVs individuales (150 archivos) y verifica contra valores del manuscrito.
2. **Fase 7 run.ps1**: `uv.exe` no estaba en PATH y `dvc status` usaba sintaxis incorrecta. Corregido.
3. **Sin errores nuevos**: validate_fase1.py, pipeline, y compilación pasan sin errores.

### Estado de artefactos

| Entregable | Archivo | OK |
|-----------|---------|----|
| Manuscrito | `main.pdf` (32 pp, 1.55 MB) | ✅ |
| Cover letter | `cover_letter.pdf` (2 pp, 114 KB) | ✅ |
| Pipeline | `02_HERRAMIENTAS/scientific-pipeline/` | ✅ |
| Código fuente | `src/`, `scripts/`, `config/` | ✅ |
| Datos | `data/raw/` (1020 CSVs), `data/patches/` (29 .npy) | ✅ |

---

## Última Sesión — Corrección Batería 222 Wh y 600 Iteraciones

**Fecha**: 2026-06-07 (Sesión 4)
**Logro**: Corregida inconsistencia numérica Fig 4 vs Tabla 3 (battery_capacity 120/180→222 Wh), re-ejecutadas 90 simulaciones con 600 iteraciones, actualizados todos los valores del paper.

### Causa raíz
- `src/agent.py` tenía `battery_capacity` en 120/180 Wh (placeholders incorrectos)
- `analyze_results.py` agregaba datos de 30 escenarios (synthetic + 29 parches), elevando HYBRID TCR a ~0.337
- Foxtech THEA-160 con TATTU 6S 10 Ah = 222 Wh

### Correcciones aplicadas

| Ítem | Antes | Después |
|------|-------|---------|
| `src/agent.py` battery_capacity | 120/180 Wh | **222 Wh** (todos los roles) |
| Iteraciones | 300 | **600** (ACO/HYBRID agotan batería en ~400) |
| Fuente figuras | 30 escenarios (synthetic + parches) | **synthetic-only** |
| n | 30 seeds × 3 configs × 300 iters | **30 seeds × 3 configs × 600 iters** |

### Valores finales

| Métrica | PSO | ACO | HYBRID |
|---------|-----|-----|--------|
| TCR | 0.311±0.092 | 0.063±0.030 | 0.298±0.051 |
| Energía residual (Wh) | 141.3±33.7 | 0.0±0.0 | 0.0±0.0 |
| σ_Energy | 66.49 | 0.00 | 0.00 |
| Área efectiva (ha) | 3.78 | 0.77 | 3.62 |
| Eficiencia (ha/Wh) | 0.0467 | 0.0035 | 0.0163 |

- HYBRID vs PSO: Mann-Whitney U=187.0, p=0.571 (no significativo — equivalencia)
- ANOVA: F₂,₈₇=118.80, p=2.22e-23 (diferencias globales sí)
- Ratio HYBRID/ACO: **4.7×** área cubierta

### Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `src/agent.py` | battery_capacity 120/180 → 222 Wh |
| `analyze_results.py` | Aislado a synthetic-only para figuras |
| `figuras/boxplot_tcr_final.png` | Regenerada (synthetic, 30 seeds, 600 iters) |
| `figuras/convergence_tcr.png` | Regenerada |
| `figuras/boxplot_sigma_energy_final.png` | Regenerada |
| `figuras/convergence_energy.png` | Regenerada |
| `main.tex` | Abstract + highlights con nuevos valores |
| `04_resultados.tex` | Tablas 3 y 4, texto ANOVAs actualizados |
| `05_discusion.tex` | U, p, sigma, 4.7× actualizados |
| `06_conclusiones.tex` | Valores finales actualizados |
| `verdad.yaml` | n_iterations 300→600, TCRs, F, p actualizados |

### Estado de artefactos

| Entregable | Archivo | OK |
|-----------|---------|----|
| Manuscrito | `main.pdf` (32 pp, 1.5 MB, 0 errores) | ✅ |
| Validación cruzada | 0 ocurrencias de valores viejos (34.61, 435.5, 150.35) en .tex | ✅ |
| Pipeline | `verdad.yaml` actualizado con nuevos TCRs | ✅ |

---

## Última Sesión — Revisión COMPAG Crítica (R1–R5) + PSO Canónico en Tránsito + Escalabilidad Sintética

**Fecha**: 2026-06-08/09 (Sesión 5)
**Logro**: Implementados todos los requisitos del dictamen REVISIONES MAYORES de COMPAG.

### Requisitos implementados

| Requisito | Descripción | Estado |
|-----------|-------------|--------|
| R1 | Flota heterogénea validada: HYBRID_HETERO como config separada (6 monitor + 4 sprayer) | ✅ |
| R2 | Tránsito híbrido ahora usa PSO canónico (w, c1, c2, r1, r2) en vez de vector dirección determinista | ✅ |
| R3 | Tracking de modo tránsito vs explotación por iteración (columnas n_transit/n_exploit en CSVs) | ✅ |
| R4 | Reclamos Lyapunov moderados a monotonicidad + conjetura ISS explícitamente señalada | ✅ |
| R5 | Escalabilidad re-ejecutada en escenario sintético (no parche real) | ✅ |

### Datos finales (n=30, iter 599, 222 Wh, synthetic)

| Config | TCR | Energy (Wh) | σ_Energy |
|--------|-----|-------------|----------|
| Boustrophedon | 1.000 ± 0.000 | 36.1 ± 0.1 | 6.14 ± 0.07 |
| PSO Multi-Niche | 0.477 ± 0.070 | 42.6 ± 27.8 | 66.35 ± 32.97 |
| HYBRID (homogéneo) | **0.333 ± 0.034** | 0.0 ± 0.0 | 0.00 ± 0.00 |
| PSO (canónico) | 0.311 ± 0.092 | 141.2 ± 33.7 | 66.49 ± 21.79 |
| HYBRID (heterogéneo) | 0.303 ± 0.038 | 0.0 ± 0.0 | 0.00 ± 0.00 |
| ACO | 0.063 ± 0.030 | 0.0 ± 0.0 | 0.00 ± 0.00 |

### Escalabilidad sintética (n=30, 600 iters)

| N | TCR |
|---|-----|
| 5 | 0.177 ± 0.029 |
| 10 | 0.333 ± 0.034 |
| 20 | 0.515 ± 0.040 |
| 30 | 0.672 ± 0.051 |

Rendimientos decrecientes: 5→10 ×1.88, 10→20 ×1.55, 20→30 ×1.31.

### Estadísticas clave
- HYBRID vs PSO: U=522, p=0.29, r=0.160 **(equivalente)**
- HYBRID vs HYBRID_HETERO: U=648, p=0.0034, r=0.441 **(significativo: homogéneo > heterogéneo)**
- ANOVA: F₅,₁₇₄=1010.37, p<10⁻¹²⁵ (diferencias globales)
- Tránsito vs explotación: >99% del tiempo en modo explotación tras t=99 iters

### Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `src/algorithms.py` | HybridPSOACO: tránsito PSO canónico (w, c1, c2, r1, r2) + tracking transit/exploit |
| `run_experiments.py` | Soporte HYBRID_HETERO + columnas n_transit/n_exploit |
| `scripts/baseline_comparison.py` | Actualizado con 6 configs + figuras |
| `scripts/plot_transit_exploit.py` | Nueva figura transit/exploit |
| `contenido/04_resultados.tex` | Tabla 6 configs, escalabilidad sintética con 30 seeds/600 iters |
| `contenido/05_discusion.tex` | Heterogeneidad ahora significativa (p=0.0034); tránsito PSO canónico |
| `contenido/06_conclusiones.tex` | Hallazgos actualizados |
| `main.tex` | Abstract + highlights con HYBRID 0.333 |
| `data/scalability_synthetic/` | N=5,10,20,30 con 30 seeds cada uno |

### Estado de artefactos

| Entregable | Archivo | OK |
|-----------|---------|----|
| Manuscrito | `main.pdf` (35 pp, 523 KB) | ✅ |
| Validación | `validate_fase1.py` (22/22 checks) | ✅ |
| Datos principales | `data/raw/synthetic/` (6×30=180 CSVs) | ✅ |
| Escalabilidad | `data/scalability_synthetic/` (4×30=120 CSVs) | ✅ |
| Análisis baselines | `data/baseline_comparison/` (6 figs, 2 CSVs) | ✅ |
| Transit/exploit fig | `figuras/transit_exploit_breakdown.pdf` | ✅ |

**Próximo paso**: Completar response letter al editor de COMPAG con tabla de cambios R1–R5 y re-someter.

---

## Última Sesión — Response Letter COMPAG + R5 (AgriSwarm-Bench reproducible)

**Fecha**: 2026-08-12 (Sesión 6)
**Logro**: Completada la response letter R1–R5 del dictamen COMPAG y cerrada la obligatoria R5 pendiente.

### Verificación de configuraciones (código + datos)

| Config | Datos | Estado |
|--------|-------|--------|
| PSO (canónico w=0.7, c1=c2=1.5, r1/r2) | 30 CSVs | ✅ |
| ACO | 30 CSVs | ✅ |
| HYBRID (homogéneo) | 30 CSVs | ✅ |
| HYBRID_HETERO (R1: 6 monitor + 4 sprayer) | 30 CSVs | ✅ |
| PSO_MULTI_NICHE (R2) | 30 CSVs | ✅ |
| BOUSTROPHEDON | 30 CSVs | ✅ |
| Escalabilidad sintética N∈{5,10,20,30} | 120 CSVs | ✅ |

### Hallazgo: discrepancia de numeración R5
- La "R5" del checklist anterior (escalabilidad sintética) corresponde en el dictamen a la **sugerencia S2**, no a la obligatoria R5.
- La obligatoria **R5** (describir AgriSwarm-Bench como componente autónomo y reproducible) estaba **pendiente**. Resuelta: README reescrito con dependencias, estructura, formatos de escenario NDVI, API de registro de algoritmos e instrucciones de reproducción figura por figura.

### Entregables de la sesión

| Entregable | Archivo | OK |
|-----------|---------|----|
| Response Letter (R1–R5 + S1–S5) | `response_letter.tex` + `response_letter.pdf` (2 pp) | ✅ |
| README AgriSwarm-Bench reproducible | `01_PROYECTO/AgriSwarm-Py/README.md` | ✅ |
| Compilación xelatex | 0 errores, 119 KB | ✅ |

### Pendiente para re-someter
- Subir README actualizado a GitHub (repo AgriSwarm-Bench)
- Validar pipeline: `.\01_PROYECTO\run.ps1 -Target all`
- Re-someter en el portal de COMPAG (S-26-07551)
