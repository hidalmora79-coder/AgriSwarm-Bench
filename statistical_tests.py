"""
statistical_tests.py

Pruebas estadísticas por escenario: Kruskal-Wallis + Mann-Whitney U post-hoc.
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu

CONFIGS = ["PSO", "ACO", "HYBRID"]
COMPARISONS = [("HYBRID", "PSO"), ("HYBRID", "ACO")]


def load_scenario_data(scenario_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(scenario_dir / "*.csv")))
    if not files:
        return None
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def compute_effect_size(x, y) -> float:
    """Cliff's Delta: non-parametric effect size for ordinal data."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    greater = 0.0
    for v1 in x:
        greater += np.sum(v1 > y)
    for v2 in y:
        greater += np.sum(x > v2)
    delta = (greater / (n1 * n2)) - 1.0
    return float(delta)


def run_tests(scenario_name: str, df: pd.DataFrame):
    final = df[df["iteration"] == df["iteration"].max()]
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")

    # 1. Kruskal-Wallis (global)
    groups = [final.loc[final["config"] == c, "tcr"].values for c in CONFIGS]
    kw_stat, kw_p = kruskal(*groups)
    sig = "YES" if kw_p < 0.05 else "NO"
    print(f"\nKruskal-Wallis (TCR final): H = {kw_stat:.4f}, p = {kw_p:.6e} [{sig}]")

    for metric, col in [("TCR", "tcr"), ("Sigma Energy", "sigma_energy")]:
        print(f"\n--- {metric} ---")
        kw_stat_m, kw_p_m = kruskal(*[final.loc[final["config"] == c, col].values for c in CONFIGS])
        sig_kw = "YES" if kw_p_m < 0.05 else "NO"
        print(f"  Kruskal-Wallis: H = {kw_stat_m:.4f}, p = {kw_p_m:.6e} [{sig_kw}]")

        for c1, c2 in COMPARISONS:
            a = final.loc[final["config"] == c1, col].values
            b = final.loc[final["config"] == c2, col].values
            u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
            delta = compute_effect_size(a, b)
            m1, s1 = np.mean(a), np.std(a)
            m2, s2 = np.mean(b), np.std(b)
            sig_mw = "YES" if u_p < 0.05 else "NO"
            print(f"  {c1} vs {c2}: MW U = {u_stat:.1f}, p = {u_p:.6e} [{sig_mw}], "
                  f"Cliff's Delta = {delta:.4f}")
            print(f"    {c1}: mean={m1:.4f} std={s1:.4f} | {c2}: mean={m2:.4f} std={s2:.4f}")


def main():
    print("=== PRUEBAS ESTADISTICAS ===")
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        print("No data/raw/ directory found. Run experiments first.")
        return

    scenarios = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    if not scenarios:
        print("No scenario subdirectories found in data/raw/")
        return

    for sc in scenarios:
        sc_dir = raw_dir / sc
        df = load_scenario_data(sc_dir)
        if df is None:
            print(f"Skip {sc}: no data")
            continue
        run_tests(sc, df)


if __name__ == "__main__":
    main()
