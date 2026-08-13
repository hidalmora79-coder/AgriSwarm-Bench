"""analyze_review_results.py

Comprehensive analysis of new experiment results after review changes.
Includes BOUSTROPHEDON, PSO_MULTI_NICHE, HYBRID (homogeneous), HYBRID_HETERO.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import shapiro, mannwhitneyu

root = Path("data/raw/synthetic")
configs = ["PSO", "ACO", "HYBRID", "HYBRID_HETERO", "BOUSTROPHEDON", "PSO_MULTI_NICHE"]
labels = {
    "PSO": "PSO",
    "ACO": "ACO",
    "HYBRID": "HYBRID (homogeneous)",
    "HYBRID_HETERO": "HYBRID (heterogeneous)",
    "BOUSTROPHEDON": "Boustrophedon",
    "PSO_MULTI_NICHE": "PSO Multi-Niche",
}
results = {}

for cfg in configs:
    prefix = cfg.lower().replace("-", "_")
    paths = sorted(root.glob(f"{prefix}_seed_*.csv"))
    if not paths:
        print(f"WARN: No data found for {cfg}")
        continue
    dfs = [pd.read_csv(p) for p in paths]
    combined = pd.concat(dfs, ignore_index=True)
    final = combined[combined["iteration"] == combined["iteration"].max()]
    results[cfg] = {"full": combined, "final": final}

    tcr = final["tcr"]
    energy = final["mean_energy"]
    sigma = final["sigma_energy"]
    print(f"\n{labels.get(cfg, cfg)}:")
    print(f"  TCR = {tcr.mean():.4f} +/- {tcr.std():.4f}")
    print(f"  Energy = {energy.mean():.1f} +/- {energy.std():.1f}")
    print(f"  Sigma_E = {sigma.mean():.3f} +/- {sigma.std():.3f}")
    if len(tcr) > 3:
        _, sw = shapiro(tcr)
        print(f"  Shapiro p = {sw:.4f}")

active = [c for c in configs if c in results]
print("\n\n--- Mann-Whitney pairwise TCR ---")
for a in active:
    for b in active:
        if a >= b:
            continue
        u, p = mannwhitneyu(results[a]["final"]["tcr"], results[b]["final"]["tcr"], alternative="two-sided")
        r = 1 - (2 * u) / (len(results[a]["final"]) * len(results[b]["final"]))
        print(f"  {labels.get(a,a):>25s} vs {labels.get(b,b):<30s}: U={u:.0f}, p={p:.4e}, r={abs(r):.3f}")

# Transit vs exploitation for HYBRID variants
for cfg in ["HYBRID", "HYBRID_HETERO"]:
    if cfg in results and "n_transit" in results[cfg]["full"].columns:
        print(f"\n--- Transit vs Exploitation ({labels.get(cfg, cfg)}) ---")
        full = results[cfg]["full"]
        final_it = full["iteration"].max()
        for label, it in [("t=0", 0), ("t=99", 99), ("t=299", 299), ("t=599", 599)]:
            subset = full[full["iteration"] == it]
            if len(subset) > 0:
                print(f"  {label}: transit={subset['n_transit'].mean():.1f} exploit={subset['n_exploit'].mean():.1f}")

print("\n\n--- Final summary table ---")
print(f"{'Config':30s} {'TCR':>10s} {'Energy':>10s} {'Sigma_E':>10s}")
print("-" * 65)
for cfg in active:
    f = results[cfg]["final"]
    tcr = f["tcr"]
    energy = f["mean_energy"]
    sigma = f["sigma_energy"]
    print(f"{labels.get(cfg, cfg):30s} {tcr.mean():.4f}+-{tcr.std():.4f} {energy.mean():.1f}+-{energy.std():.1f} {sigma.mean():.3f}+-{sigma.std():.3f}")
