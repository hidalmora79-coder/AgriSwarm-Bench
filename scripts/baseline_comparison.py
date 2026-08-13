"""baseline_comparison.py

Comparative analysis: PSO, ACO, HYBRID (homo), HYBRID_HETERO,
BOUSTROPHEDON, PSO_MULTI_NICHE.
Generates figures + statistical tables for revised manuscript.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import shapiro, mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "synthetic"
OUT = ROOT / "data" / "baseline_comparison"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = ["HYBRID", "HYBRID_HETERO", "PSO", "ACO", "PSO_MULTI_NICHE", "BOUSTROPHEDON"]
COLORS = {
    "HYBRID": "#2ECC71",
    "HYBRID_HETERO": "#27AE60",
    "PSO": "#3498DB",
    "ACO": "#E74C3C",
    "PSO_MULTI_NICHE": "#9B59B6",
    "BOUSTROPHEDON": "#F39C12",
}
LABELS = {
    "HYBRID": "HYBRID",
    "HYBRID_HETERO": "HYBRID-Hetro",
    "PSO": "PSO",
    "ACO": "ACO",
    "PSO_MULTI_NICHE": "PSO Multi-Niche",
    "BOUSTROPHEDON": "Boustrophedon",
}


def load_all() -> dict:
    data = {}
    for cfg in CONFIGS:
        prefix = cfg.lower().replace("-", "_")
        paths = sorted(RAW.glob(f"{prefix}_seed_*.csv"))
        if not paths:
            print(f"WARN: No data for {cfg}")
            continue
        dfs = [pd.read_csv(p) for p in paths]
        combined = pd.concat(dfs, ignore_index=True)
        final = combined[combined["iteration"] == combined["iteration"].max()]
        data[cfg] = {"full": combined, "final": final}
    return data


def compute_stats(data: dict) -> pd.DataFrame:
    rows = []
    for cfg in CONFIGS:
        if cfg not in data:
            continue
        final = data[cfg]["final"]
        tcr = final["tcr"]
        energy = final["mean_energy"]
        sigma = final["sigma_energy"]
        _, sw_tcr = shapiro(tcr)
        rows.append({
            "Config": cfg,
            "TCR_mean": f"{tcr.mean():.4f}",
            "TCR_std": f"{tcr.std():.4f}",
            "Energy_mean": f"{energy.mean():.1f}",
            "Energy_std": f"{energy.std():.1f}",
            "Sigma_mean": f"{sigma.mean():.3f}",
            "Shapiro_p": f"{sw_tcr:.4f}",
        })
    return pd.DataFrame(rows)


def mannwhitney_pairwise(data: dict, metric: str = "tcr"):
    from itertools import combinations
    results = []
    active = [c for c in CONFIGS if c in data]
    for a, b in combinations(active, 2):
        u, p = mannwhitneyu(data[a]["final"][metric], data[b]["final"][metric],
                            alternative="two-sided")
        r = 1 - (2 * u) / (len(data[a]["final"]) * len(data[b]["final"]))
        results.append({
            "A": a, "B": b,
            "U": f"{u:.0f}", "p": f"{p:.4e}",
            "r": f"{abs(r):.3f}",
        })
    return pd.DataFrame(results).sort_values("p")


def plot_convergence(data: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    active = [c for c in CONFIGS if c in data]
    for cfg in active:
        grouped = data[cfg]["full"].groupby("iteration")["tcr"].agg(["mean", "std"]).reset_index()
        it = grouped["iteration"]
        mean = grouped["mean"]
        std = grouped["std"]
        axes[0].plot(it, mean, label=LABELS[cfg], color=COLORS[cfg], linewidth=1.5)
        axes[0].fill_between(it, mean - std, mean + std, alpha=0.15, color=COLORS[cfg])

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("TCR")
    axes[0].set_title("TCR Convergence (mean \u00b1 std, n=30)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    for cfg in active:
        grouped = data[cfg]["full"].groupby("iteration")["mean_energy"].agg(["mean", "std"]).reset_index()
        it = grouped["iteration"]
        mean = grouped["mean"]
        std = grouped["std"]
        axes[1].plot(it, mean, label=LABELS[cfg], color=COLORS[cfg], linewidth=1.5)
        axes[1].fill_between(it, mean - std, mean + std, alpha=0.15, color=COLORS[cfg])

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean Residual Energy (Wh)")
    axes[1].set_title("Energy Decay (mean \u00b1 std, n=30)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT / "baseline_convergence.png"
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_boxplots(data: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    active = [c for c in CONFIGS if c in data]
    bp_data_tcr = [data[cfg]["final"]["tcr"].values for cfg in active]
    bp_data_energy = [data[cfg]["final"]["mean_energy"].values for cfg in active]
    tick_labels = [LABELS[c] for c in active]

    bp1 = axes[0].boxplot(bp_data_tcr, tick_labels=tick_labels,
                          patch_artist=True, widths=0.6)
    for patch, color in zip(bp1["boxes"], [COLORS[c] for c in active]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_ylabel("TCR (final iteration)")
    axes[0].set_title("TCR Distribution by Algorithm")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=15)

    bp2 = axes[1].boxplot(bp_data_energy, tick_labels=tick_labels,
                          patch_artist=True, widths=0.6)
    for patch, color in zip(bp2["boxes"], [COLORS[c] for c in active]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel("Mean Residual Energy (Wh)")
    axes[1].set_title("Energy Distribution by Algorithm")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = OUT / "baseline_boxplots.png"
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_efficiency_tradeoff(data: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    active = [c for c in CONFIGS if c in data]
    for cfg in active:
        final = data[cfg]["final"]
        tcr_mean = final["tcr"].mean()
        tcr_std = final["tcr"].std()
        energy_mean = final["mean_energy"].mean()
        energy_std = final["mean_energy"].std()
        ax.errorbar(energy_mean, tcr_mean,
                    xerr=energy_std, yerr=tcr_std,
                    fmt="o", color=COLORS[cfg], capsize=5, capthick=1.5,
                    markersize=10, label=LABELS[cfg])

    ax.set_xlabel("Mean Residual Energy (Wh) \u2193 better")
    ax.set_ylabel("TCR \u2191 better")
    ax.set_title("TCR-Energy Tradeoff (n=30, iteration 599)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    path = OUT / "baseline_tradeoff.png"
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    print("Loading experiment data...")
    data = load_all()

    print("\nComputing statistics...")
    stats = compute_stats(data)
    print(stats.to_string(index=False))

    stats_path = OUT / "baseline_stats.csv"
    stats.to_csv(stats_path, index=False)
    print(f"\nSaved: {stats_path}")

    print("\nPairwise Mann-Whitney comparisons (TCR):")
    pw = mannwhitney_pairwise(data, "tcr")
    print(pw.to_string(index=False))
    pw_path = OUT / "baseline_pairwise.csv"
    pw.to_csv(pw_path, index=False)
    print(f"Saved: {pw_path}")

    print("\nGenerating figures...")
    plot_convergence(data)
    plot_boxplots(data)
    plot_efficiency_tradeoff(data)

    print(f"\nDone. Output: {OUT}/")


if __name__ == "__main__":
    main()
