"""Calculate statistics for manuscript revision."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import f_oneway, kruskal

root = Path("data/raw/synthetic")
CONFIGS = ["HYBRID", "HYBRID_HETERO", "PSO", "ACO", "PSO_MULTI_NICHE", "BOUSTROPHEDON"]
labels = {"HYBRID": "HYBRID (homo)", "HYBRID_HETERO": "HYBRID (hetero)",
          "PSO": "PSO", "ACO": "ACO", "PSO_MULTI_NICHE": "PSO Multi-Niche",
          "BOUSTROPHEDON": "Boustrophedon"}

tcr_data = []
energy_data = []
sigma_data = []
group_names = []

for cfg in CONFIGS:
    prefix = cfg.lower().replace("-", "_")
    paths = sorted(root.glob(f"{prefix}_seed_*.csv"))
    dfs = [pd.read_csv(p) for p in paths]
    combined = pd.concat(dfs, ignore_index=True)
    final = combined[combined["iteration"] == combined["iteration"].max()]
    tcr_data.append(final["tcr"].values)
    energy_data.append(final["mean_energy"].values)
    sigma_data.append(final["sigma_energy"].values)
    group_names.append(labels[cfg])

# ANOVA on all 6 groups
f_stat, p_val = f_oneway(*tcr_data)
print(f"\nANOVA (all 6 groups): F = {f_stat:.2f}, p = {p_val:.4e}")

# ANOVA on 3 core: HYBRID, PSO, ACO
core = [tcr_data[0], tcr_data[2], tcr_data[3]]
f_core, p_core = f_oneway(*core)
print(f"ANOVA (HYBRID, PSO, ACO): F = {f_core:.2f}, p = {p_core:.4e}")

# ANOVA on 5 swarm-only (excl Boustrophedon)
swarm = tcr_data[:5]
f_swarm, p_swarm = f_oneway(*swarm)
print(f"ANOVA (5 swarm): F = {f_swarm:.2f}, p = {p_swarm:.4e}")

# Kruskal-Wallis
h_stat, kw_p = kruskal(*tcr_data)
print(f"\nKruskal-Wallis (all 6): H = {h_stat:.2f}, p = {kw_p:.4e}")

# Per-group stats
print(f"\n{'Config':25s} {'TCR':>12s} {'Energy':>12s} {'Sigma_E':>12s}")
print("-" * 65)
for i, cfg in enumerate(CONFIGS):
    t = tcr_data[i]
    e = energy_data[i]
    s = sigma_data[i]
    print(f"{labels[cfg]:25s} {t.mean():.4f}+-{t.std():.4f}  {e.mean():.1f}+-{e.std():.1f}  {s.mean():.3f}+-{s.std():.3f}")

# Area metrics
print("\n\nAgricultural metrics:")
for i, cfg in enumerate(CONFIGS):
    t = tcr_data[i]
    e = energy_data[i]
    area = t.mean() * 12.15
    area_eff = area / (222.0 - e.mean()) if e.mean() < 222 else 0.0
    print(f"{labels[cfg]:25s} Area={area:.2f} ha, Efficiency={area_eff:.4f} ha/Wh")

# Transit ratio for hybrid variants
for cfg_name, cfg_key in [("HYBRID (homo)", "HYBRID"), ("HYBRID (hetero)", "HYBRID_HETERO")]:
    prefix = cfg_key.lower().replace("-", "_")
    paths = sorted(root.glob(f"{prefix}_seed_*.csv"))
    dfs = [pd.read_csv(p) for p in paths]
    combined = pd.concat(dfs, ignore_index=True)
    if "n_transit" in combined.columns:
        final_it = combined["iteration"].max()
        total_transit = combined.groupby("iteration")["n_transit"].mean()
        total_exploit = combined.groupby("iteration")["n_exploit"].mean()
        avg_transit_pct = (total_transit.mean() / (total_transit.mean() + total_exploit.mean())) * 100
        print(f"\n{cfg_name}: avg transit = {total_transit.mean():.1f} agents ({avg_transit_pct:.1f}%)")
        # First 100 iters
        early = combined[combined["iteration"] <= 100]
        early_t = early.groupby("iteration")["n_transit"].mean().mean()
        early_e = early.groupby("iteration")["n_exploit"].mean().mean()
        print(f"  First 100 iters: transit={early_t:.1f}, exploit={early_e:.1f}")
