"""validate_fase1.py

Validación integral de datos y experimentos Fase 1–4.
Verifica integridad de archivos, consistencia de CSV y métricas clave.
Útil como pre-commit hook y para depuración rápida.
"""

import sys, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PATCHES = ROOT / "data" / "patches"
METRICS = ROOT / "data" / "agricultural_metrics"
SCALABILITY = ROOT / "data" / "scalability"
SENSITIVITY = ROOT / "data" / "sensitivity"


def check(condition: bool, msg: str, fatal: bool = False):
    status = "PASS" if condition else "FAIL"
    icon = "+" if condition else "-"
    print(f"  [{icon}] {msg}")
    if not condition and fatal:
        print(f"\n  FATAL: {msg} - aborting.")
        sys.exit(1)
    return condition


def validate_patches():
    print("\n--- FASE 1: PARCHES SENTINEL-2 ---")
    npy_files = sorted(PATCHES.glob("*.npy"))
    meta = PATCHES / "patches_metadata.csv"
    check(meta.exists(), f"patches_metadata.csv exists")
    check(len(npy_files) == 29, f"29 .npy files (found {len(npy_files)})")
    if meta.exists():
        df = pd.read_csv(meta)
        check("# name" in df.columns, "metadata has '# name' column")
        n_classes = df["stratum"].nunique() if "stratum" in df.columns else 0
        check(n_classes >= 4, f"metadata has {n_classes} strata")
    for f in npy_files:
        arr = np.load(f)
        shape_ok = arr.shape == (100, 100)
        if not shape_ok:
            check(False, f"{f.name}: shape={arr.shape} (expected 100×100)")
            break
    else:
        check(True, "all 29 patches are 100×100")


def validate_synthetic():
    print("\n--- FASE 2A: ESCENARIO SINTETICO ---")
    syn = RAW / "synthetic"
    csvs = sorted(syn.glob("*.csv"))
    total = len(csvs)
    check(total == 150, f"150 CSV files (5 configs x 30 seeds, found {total})")
    if total == 0:
        return
    df = pd.read_csv(csvs[0])
    expected_cols = {"seed", "config", "scenario", "iteration", "tcr", "mean_energy", "sigma_energy"}
    check(expected_cols.issubset(df.columns), f"columns: {sorted(df.columns)}")
    check(df["iteration"].max() >= 299, f"max iteration >= 299 (found {df['iteration'].max()})")


def validate_real_patches():
    print("\n--- FASE 2B: PARCHES REALES ---")
    patch_dirs = [d for d in RAW.iterdir() if d.is_dir() and d.name != "synthetic"]
    check(len(patch_dirs) == 29, f"29 patch directories (found {len(patch_dirs)})")
    all_ok = True
    for d in patch_dirs:
        n_csv = len(list(d.glob("*.csv")))
        if n_csv != 30:
            check(False, f"  {d.name}: {n_csv} CSVs (expected 30)")
            all_ok = False
    if all_ok:
        check(True, "all 29 dirs have 30 CSVs each")


def validate_scalability():
    print("\n--- FASE 2C: ESCALABILIDAD ---")
    csv = SCALABILITY / "scalability_results.csv"
    img = SCALABILITY / "scalability_tcr_vs_n.png"
    check(csv.exists(), f"results CSV exists")
    check(img.exists(), f"figure exists")
    if csv.exists():
        df = pd.read_csv(csv)
        n_fleet_sizes = df["n_drones"].nunique() if "n_drones" in df.columns else 0
        check(n_fleet_sizes >= 4, f"fleet sizes: {n_fleet_sizes} (expected 4)")


def validate_sensitivity():
    print("\n--- FASE 2D: SENSIBILIDAD ---")
    csv = SENSITIVITY / "sensitivity_results.csv"
    img = SENSITIVITY / "sensitivity_heatmap.png"
    check(csv.exists(), f"results CSV exists")
    check(img.exists(), f"heatmap figure exists")
    if csv.exists():
        df = pd.read_csv(csv)
        n_rho = df["rho"].nunique() if "rho" in df.columns else 0
        n_beta = df["beta"].nunique() if "beta" in df.columns else 0
        check(n_rho >= 3, f"rho values: {n_rho} (expected 3)")
        check(n_beta >= 3, f"beta values: {n_beta} (expected 3)")


def validate_agricultural_metrics():
    print("\n--- FASE 4: METRICAS AGRICOLAS ---")
    expected = [
        "effective_area.png", "efficiency_ha_per_wh.png",
        "partial_tcr_progression.png", "zone_coverage_progression.png",
        "summary_metrics.csv", "zone_coverage.csv",
    ]
    for fname in expected:
        check((METRICS / fname).exists(), f"{fname} exists")


def validate_baseline_comparison():
    print("\n--- FASE 5: BASELINES ---")
    bc = ROOT / "data" / "baseline_comparison"
    expected = [
        "baseline_stats.csv", "baseline_pairwise.csv",
        "baseline_convergence.png", "baseline_boxplots.png", "baseline_tradeoff.png",
    ]
    for fname in expected:
        check((bc / fname).exists(), f"{fname} exists")
    if (bc / "baseline_stats.csv").exists():
        df = pd.read_csv(bc / "baseline_stats.csv")
        n_configs = len(df)
        check(n_configs == 5, f"5 configs in stats table (found {n_configs})")


def validate_script_exists():
    print("\n--- SCRIPTS ---")
    scripts_dir = ROOT / "scripts"
    expected = [
        "agricultural_metrics.py", "process_sentinel2.py",
        "figures_publication.py", "validate_fase1.py",
        "baseline_comparison.py",
    ]
    for s in expected:
        check((scripts_dir / s).exists(), f"{s} exists")


def main():
    print("=" * 50)
    print("  VALIDACION INTEGRAL - AgriSwarm-Py")
    print("=" * 50)
    validate_patches()
    validate_synthetic()
    validate_real_patches()
    validate_scalability()
    validate_sensitivity()
    validate_agricultural_metrics()
    validate_baseline_comparison()
    validate_script_exists()
    print("\n" + "=" * 50)
    print("  VALIDACION COMPLETADA")
    print("=" * 50)


if __name__ == "__main__":
    main()
