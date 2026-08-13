"""
analyze_results.py

Genera figuras (300 DPI) por escenario a partir de los resultados Monte Carlo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob
import os
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "grid.alpha": 0.3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

DPI = 300
FIG_SIZE = (8, 5.5)

CONFIG_LABELS = {
    "PSO": "PSO (Baseline)",
    "ACO": "ACO (Baseline)",
    "HYBRID": "Hybrid PSO-ACO",
}

CONFIG_COLORS = {
    "PSO": "#1f77b4",
    "ACO": "#ff7f0e",
    "HYBRID": "#2ca02c",
}


def load_scenario_data(scenario_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(scenario_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {scenario_dir}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def plot_tcr_boxplot(df, scenario_name, output_dir):
    final = df[df["iteration"] == df["iteration"].max()].copy()
    final["config_label"] = final["config"].map(CONFIG_LABELS)
    order = [CONFIG_LABELS[c] for c in ["PSO", "ACO", "HYBRID"]]
    data_by_config = [final.loc[final["config_label"] == lbl, "tcr"].values for lbl in order]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bp = ax.boxplot(data_by_config, tick_labels=order, patch_artist=True,
                    widths=0.55, medianprops=dict(color="black", linewidth=1.8),
                    flierprops=dict(marker="o", markersize=4, alpha=0.6, color="gray"))

    for patch, color in zip(bp["boxes"], [CONFIG_COLORS[c] for c in ["PSO", "ACO", "HYBRID"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    for w in bp["whiskers"]:
        w.set(color="black", linewidth=1.4)
    for c in bp["caps"]:
        c.set(color="black", linewidth=1.4)

    ax.set_ylabel("Weighted Coverage Rate (TCR)")
    ax.set_xlabel("Algorithm")
    ax.set_title(f"Final TCR Distribution — {scenario_name}", pad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)

    for i, (lbl, vals) in enumerate(zip(order, data_by_config), start=1):
        m, s = np.mean(vals), np.std(vals)
        ax.text(i, 0.95, f"$\\mu$={m:.4f}\n$\\sigma$={s:.4f}",
                ha="center", va="top", transform=ax.get_xaxis_transform(),
                fontsize=9.5, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))

    fig.savefig(output_dir / f"boxplot_tcr_{scenario_name}.png", dpi=DPI)
    plt.close(fig)
    print(f"  Saved: boxplot_tcr_{scenario_name}.png")


def plot_energy_boxplot(df, scenario_name, output_dir):
    final = df[df["iteration"] == df["iteration"].max()].copy()
    final["config_label"] = final["config"].map(CONFIG_LABELS)
    order = [CONFIG_LABELS[c] for c in ["PSO", "ACO", "HYBRID"]]
    data_by_config = [final.loc[final["config_label"] == lbl, "sigma_energy"].values for lbl in order]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bp = ax.boxplot(data_by_config, tick_labels=order, patch_artist=True,
                    widths=0.55, medianprops=dict(color="black", linewidth=1.8),
                    flierprops=dict(marker="o", markersize=4, alpha=0.6, color="gray"))

    for patch, color in zip(bp["boxes"], [CONFIG_COLORS[c] for c in ["PSO", "ACO", "HYBRID"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    for w in bp["whiskers"]:
        w.set(color="black", linewidth=1.4)
    for c in bp["caps"]:
        c.set(color="black", linewidth=1.4)

    ax.set_ylabel("Energy Std. Dev. (units)")
    ax.set_xlabel("Algorithm")
    ax.set_title(f"Final Energy Dispersion — {scenario_name}", pad=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)

    for i, (lbl, vals) in enumerate(zip(order, data_by_config), start=1):
        m, s = np.mean(vals), np.std(vals)
        ax.text(i, 0.95, f"$\\mu$={m:.2f}\n$\\sigma$={s:.2f}",
                ha="center", va="top", transform=ax.get_xaxis_transform(),
                fontsize=9.5, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))

    fig.savefig(output_dir / f"boxplot_sigma_energy_{scenario_name}.png", dpi=DPI)
    plt.close(fig)
    print(f"  Saved: boxplot_sigma_energy_{scenario_name}.png")


def plot_tcr_convergence(df, scenario_name, output_dir):
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for config in ["PSO", "ACO", "HYBRID"]:
        subset = df[df["config"] == config]
        grouped = subset.groupby("iteration")["tcr"].agg(["mean", "std"])
        iters = grouped.index.values.astype(int)
        mean_tcr = grouped["mean"].values
        std_tcr = grouped["std"].values

        color = CONFIG_COLORS[config]
        ax.plot(iters, mean_tcr, label=CONFIG_LABELS[config], color=color, linewidth=2.2)
        ax.fill_between(iters, mean_tcr - std_tcr, mean_tcr + std_tcr,
                        color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average TCR")
    ax.set_title(f"TCR Convergence — {scenario_name}", pad=10)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="gray")
    ax.grid(linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    fig.savefig(output_dir / f"convergence_tcr_{scenario_name}.png", dpi=DPI)
    plt.close(fig)
    print(f"  Saved: convergence_tcr_{scenario_name}.png")


def plot_energy_convergence(df, scenario_name, output_dir):
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for config in ["PSO", "ACO", "HYBRID"]:
        subset = df[df["config"] == config]
        grouped = subset.groupby("iteration")["mean_energy"].agg(["mean", "std"])
        iters = grouped.index.values.astype(int)
        mean_e = grouped["mean"].values
        std_e = grouped["std"].values

        color = CONFIG_COLORS[config]
        ax.plot(iters, mean_e, label=CONFIG_LABELS[config], color=color, linewidth=2.2)
        ax.fill_between(iters, mean_e - std_e, mean_e + std_e,
                        color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Energy")
    ax.set_title(f"Mean Energy Consumption — {scenario_name}", pad=10)
    ax.legend(loc="best", framealpha=0.9, edgecolor="gray")
    ax.grid(linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.savefig(output_dir / f"convergence_energy_{scenario_name}.png", dpi=DPI)
    plt.close(fig)
    print(f"  Saved: convergence_energy_{scenario_name}.png")


def main():
    print("=== ANÁLISIS DE RESULTADOS MONTE CARLO ===")
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        print("No data/raw/ directory found. Run experiments first.")
        return

    scenarios = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    if not scenarios:
        print("No scenario subdirectories found in data/raw/")
        return

    print(f"Escenarios detectados: {scenarios}")
    output_dir = Path("data/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    for sc in scenarios:
        sc_dir = raw_dir / sc
        print(f"\n--- Escenario: {sc} ---")
        try:
            df = load_scenario_data(sc_dir)
        except FileNotFoundError as e:
            print(f"  Skip: {e}")
            continue

        print(f"  Records: {len(df)} | Configs: {df['config'].nunique()} | Seeds: {df['seed'].nunique()}")

        plot_tcr_boxplot(df, sc, output_dir)
        plot_energy_boxplot(df, sc, output_dir)
        plot_tcr_convergence(df, sc, output_dir)
        plot_energy_convergence(df, sc, output_dir)

    print(f"\n=== ALL FIGURES GENERATED: {output_dir} ===")


if __name__ == "__main__":
    main()
