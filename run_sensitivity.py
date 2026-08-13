"""
run_sensitivity.py

Fase 2D — Sensibilidad de hiperparámetros: diseño factorial 3×3
  ρ (pheromone evaporation) ∈ {0.05, 0.10, 0.20}
  β (pheromone influence)   ∈ {1.0, 2.0, 3.0}
Escenario sintético, HYBRID, 10 semillas × 300 iteraciones.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse
from pathlib import Path
from datetime import datetime

from src.environment import AgriculturalField
from src.agent import Drone
from src.algorithms import HybridPSOACO
from src.metrics import update_coverage, calculate_tcr_from_mask, calculate_energy_stats
from src.config import get_profile, hybrid_params, experiment_params

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

RHO_VALS = [0.05, 0.10, 0.20]
BETA_VALS = [1.0, 2.0, 3.0]


def run_single(rho: float, beta: float, seed: int, n_drones: int = 10,
               iterations: int = 300) -> pd.DataFrame:
    field = AgriculturalField(size=100, seed=seed)
    field.generate_ndvi_map()

    config = get_profile()
    hp = hybrid_params(config)
    threshold = hp.get("stress_threshold", 0.4)

    targets = HybridPSOACO.get_niche_targets(field.ndvi_map, n_drones, threshold=threshold)
    exp = experiment_params(config)
    field_size = exp.get("field_size", 100)
    start_offset = exp.get("start_offset_hybrid", 15.0)

    main_rng = np.random.default_rng(seed)
    drone_rngs = main_rng.spawn(n_drones)

    drones = []
    for i in range(n_drones):
        offset = main_rng.uniform(-start_offset, start_offset, size=2)
        start_pos = np.clip(targets[i] + offset, 0, field_size - 1)
        drones.append(Drone(i, start_pos, drone_rngs[i], role="monitor"))

    optimizer = HybridPSOACO(drones, field_size=field_size,
                              ndvi_map=field.ndvi_map, config=config,
                              rho=rho, beta=beta)

    history = []
    covered_mask = np.zeros((field_size, field_size), dtype=bool)

    for t in range(iterations):
        optimizer.step(drones, field)
        current_positions = [d.pos for d in drones]
        update_coverage(covered_mask, current_positions)
        tcr = calculate_tcr_from_mask(field.ndvi_map, covered_mask)
        mean_e, std_e = calculate_energy_stats(drones)

        history.append({
            "rho": rho,
            "beta": beta,
            "seed": seed,
            "iteration": t,
            "tcr": tcr,
            "mean_energy": mean_e,
            "sigma_energy": std_e,
        })

    return pd.DataFrame(history)


def plot_heatmap(df: pd.DataFrame, output_path: str):
    final = df[df["iteration"] == df["iteration"].max()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for idx, metric in enumerate(["tcr", "sigma_energy"]):
        ax = axes[idx]
        table = np.full((len(RHO_VALS), len(BETA_VALS)), np.nan)
        table_std = np.full((len(RHO_VALS), len(BETA_VALS)), np.nan)

        for i, rho in enumerate(RHO_VALS):
            for j, beta in enumerate(BETA_VALS):
                vals = final.loc[(final["rho"] == rho) & (final["beta"] == beta), metric]
                if len(vals) > 0:
                    table[i, j] = vals.mean()
                    table_std[i, j] = vals.std()

        im = ax.imshow(table, cmap="YlGnBu" if metric == "tcr" else "OrRd",
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(BETA_VALS)))
        ax.set_xticklabels([str(b) for b in BETA_VALS])
        ax.set_yticks(range(len(RHO_VALS)))
        ax.set_yticklabels([str(r) for r in RHO_VALS])
        ax.set_xlabel(r"$\beta$ (pheromone influence)")
        ax.set_ylabel(r"$\rho$ (evaporation rate)")

        title = r"Mean TCR" if metric == "tcr" else r"Mean $\sigma_{\mathrm{Energy}}$"
        ax.set_title(title, pad=8)

        for i in range(len(RHO_VALS)):
            for j in range(len(BETA_VALS)):
                val = table[i, j]
                std = table_std[i, j]
                txt = f"{val:.3f}\n±{std:.3f}" if metric == "tcr" else f"{val:.1f}\n±{std:.1f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8.5, fontfamily="monospace",
                        color="black" if table[i, j] < table.max() * 0.7 else "white")

        fig.colorbar(im, ax=ax, shrink=0.75)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Figura guardada: {output_path}")


def plot_boxplot(df: pd.DataFrame, output_path: str):
    final = df[df["iteration"] == df["iteration"].max()].copy()
    fig, ax = plt.subplots(figsize=(9, 5))

    labels = [f"ρ={r}\nβ={b}" for r in RHO_VALS for b in BETA_VALS]
    data = [final.loc[(final["rho"] == r) & (final["beta"] == b), "tcr"].values
            for r in RHO_VALS for b in BETA_VALS]
    colors = ["#3498DB", "#2ECC71", "#E74C3C"] * 3

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                     medianprops=dict(color="black", linewidth=1.5),
                     flierprops=dict(marker="o", markersize=4, alpha=0.5, color="gray"))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    for w in bp["whiskers"]: w.set(color="black", linewidth=1.2)
    for c in bp["caps"]: c.set(color="black", linewidth=1.2)

    ax.set_ylabel("Task Completion Rate (TCR)")
    ax.set_title("Hyperparameter Sensitivity: TCR by (ρ, β) Configuration", pad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Figura guardada: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity (Fase 2D)")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", default="data/sensitivity")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    total = len(RHO_VALS) * len(BETA_VALS) * args.seeds
    count = 0
    start_time = datetime.now()

    print("=== SENSIBILIDAD DE HIPERPARAMETROS (HYBRID) ===")
    print("rho = %s" % RHO_VALS)
    print("beta = %s" % BETA_VALS)
    print("Semillas: %d | Iteraciones: %d" % (args.seeds, args.iterations))
    print("Total corridas: %d" % total)
    print()

    for rho in RHO_VALS:
        for beta in BETA_VALS:
            print()
            print("--- rho=%.2f, beta=%.1f ---" % (rho, beta))
            for s in range(args.seeds):
                df = run_single(rho, beta, s, iterations=args.iterations)
                all_dfs.append(df)
                count += 1
                print("  [%d/%d] seed=%d - TCR = %.4f" % (count, total, s, df['tcr'].iloc[-1]))

    duration = datetime.now() - start_time
    print()
    print("FINALIZADO en %.2fs" % duration.total_seconds())
    print("Corridas: %d" % count)

    df_all = pd.concat(all_dfs, ignore_index=True)

    csv_path = output_dir / "sensitivity_results.csv"
    df_all.to_csv(csv_path, index=False)
    print("Datos guardados: %s" % csv_path)

    plot_heatmap(df_all, str(output_dir / "sensitivity_heatmap.png"))
    plot_boxplot(df_all, str(output_dir / "sensitivity_boxplot.png"))

    final = df_all[df_all["iteration"] == df_all["iteration"].max()]
    print()
    print("=== RESUMEN TCR FINAL ===")
    for rho in RHO_VALS:
        for beta in BETA_VALS:
            vals = final.loc[(final["rho"] == rho) & (final["beta"] == beta), "tcr"]
            print("  rho=%.2f, beta=%.1f:  TCR = %.4f +/- %.4f" % (rho, beta, vals.mean(), vals.std()))

    print()
    print("=== RESUMEN sigma_ENERGIA FINAL ===")
    for rho in RHO_VALS:
        for beta in BETA_VALS:
            vals = final.loc[(final["rho"] == rho) & (final["beta"] == beta), "sigma_energy"]
            print("  rho=%.2f, beta=%.1f:  sigma_E = %.2f +/- %.2f" % (rho, beta, vals.mean(), vals.std()))


if __name__ == "__main__":
    main()
