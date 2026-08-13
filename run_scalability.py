"""
run_scalability.py

Fase 2C — Escalabilidad de flota: corre HYBRID con N = 5, 10, 20, 30
agentes sobre el parche real más representativo y genera figura TCR vs N.
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
from src.config import get_profile, hybrid_params, experiment_params, build_overrides_from_args

# --- configuración global de figuras ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

FLET_SIZES = [5, 10, 20, 30]
COLORS = {5: "#3498DB", 10: "#2ECC71", 20: "#E67E22", 30: "#E74C3C"}


def run_single_n(n_drones: int, seed: int, patch_path: str,
                 iterations: int = 300, config: dict = None) -> pd.DataFrame:
    field = AgriculturalField.from_ndvi_file(patch_path, seed=seed)
    field.generate_ndvi_map()

    hp = hybrid_params(config)
    targets = HybridPSOACO.get_niche_targets(
        field.ndvi_map, n_drones, threshold=hp.get("stress_threshold", 0.4)
    )
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
                              ndvi_map=field.ndvi_map, config=config)

    history = []
    covered_mask = np.zeros((field_size, field_size), dtype=bool)

    for t in range(iterations):
        optimizer.step(drones, field)
        current_positions = [d.pos for d in drones]
        update_coverage(covered_mask, current_positions)
        tcr = calculate_tcr_from_mask(field.ndvi_map, covered_mask)
        mean_e, std_e = calculate_energy_stats(drones)

        history.append({
            "n_drones": n_drones,
            "seed": seed,
            "iteration": t,
            "tcr": tcr,
            "mean_energy": mean_e,
            "sigma_energy": std_e,
        })

    return pd.DataFrame(history)


def run_scalability_experiment(n_values: list[int], seeds: int, patch_path: str,
                                iterations: int, config: dict) -> pd.DataFrame:
    all_dfs = []
    total = len(n_values) * seeds
    count = 0
    start_time = datetime.now()

    print(f"=== ESCALABILIDAD DE FLOTA (HYBRID) ===")
    print(f"N = {n_values}")
    print(f"Semillas: {seeds} | Iteraciones: {iterations}")
    print(f"Parche: {Path(patch_path).stem}")
    print(f"Total corridas: {total}")
    print()

    for n in n_values:
        print(f"\n--- N = {n} ---")
        for s in range(seeds):
            df = run_single_n(n, s, patch_path, iterations, config)
            all_dfs.append(df)
            count += 1
            print(f"  [{count}/{total}] N={n}, seed={s} — TCR final = {df['tcr'].iloc[-1]:.4f}")

    duration = datetime.now() - start_time
    print(f"\nFINALIZADO en {duration.total_seconds():.2f}s")
    print(f"Corridas: {count}")
    return pd.concat(all_dfs, ignore_index=True)


def plot_scalability(df: pd.DataFrame, output_path: str):
    final = df[df["iteration"] == df["iteration"].max()].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- TCR vs N (boxplot + medias) ---
    n_values = sorted(final["n_drones"].unique())
    data_by_n = [final.loc[final["n_drones"] == n, "tcr"].values for n in n_values]
    means = [np.mean(d) for d in data_by_n]
    stds = [np.std(d) for d in data_by_n]

    bp = ax1.boxplot(data_by_n, tick_labels=[str(n) for n in n_values],
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=1.8),
                     flierprops=dict(marker="o", markersize=4, alpha=0.5, color="gray"))

    for patch, color in zip(bp["boxes"], [COLORS[n] for n in n_values]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    for w in bp["whiskers"]: w.set(color="black", linewidth=1.2)
    for c in bp["caps"]: c.set(color="black", linewidth=1.2)

    ax1.plot(range(1, len(n_values) + 1), means, color="black",
             marker="D", linestyle="--", linewidth=1.5, markersize=7, zorder=5)
    ax1.set_xlabel("Fleet Size ($N$)")
    ax1.set_ylabel("Task Completion Rate (TCR)")
    ax1.set_title("TCR vs Fleet Size\n(Final Iteration)", pad=10)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax1.grid(axis="y", linestyle="--", linewidth=0.5)
    ax1.set_axisbelow(True)

    # anotar medias
    for i, (m, s) in enumerate(zip(means, stds), start=1):
        ax1.text(i, 0.02, f"$\mu$={m:.3f}\n$\sigma$={s:.3f}",
                 ha="center", va="bottom", fontsize=8.5, fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor="gray", alpha=0.8))

    # --- Convergencia TCR por N ---
    for n in n_values:
        subset = df[df["n_drones"] == n]
        grouped = subset.groupby("iteration")["tcr"].agg(["mean", "std"])
        iters = grouped.index.values.astype(int)
        m = grouped["mean"].values
        s = grouped["std"].values
        ax2.plot(iters, m, label=f"N = {n}", color=COLORS[n], linewidth=2.0)
        ax2.fill_between(iters, m - s, m + s, color=COLORS[n], alpha=0.12, linewidth=0)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Average TCR")
    ax2.set_title("TCR Convergence\nby Fleet Size", pad=10)
    ax2.legend(loc="lower right", framealpha=0.9, edgecolor="gray")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax2.grid(linestyle="--", linewidth=0.5)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Figura guardada: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fleet scalability experiment (Fase 2C)")
    parser.add_argument("--patch", default="data/patches/moderado_13.npy",
                        help="Path to the representative NDVI patch")
    parser.add_argument("--seeds", type=int, default=15,
                        help="Number of Monte Carlo seeds per fleet size")
    parser.add_argument("--iterations", type=int, default=300,
                        help="Iterations per simulation")
    parser.add_argument("--n-values", type=int, nargs="+", default=FLET_SIZES,
                        help="Fleet sizes to test")
    parser.add_argument("--output", default="data/scalability",
                        help="Output directory for results and figures")
    args = parser.parse_args()

    config = get_profile()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = run_scalability_experiment(
        n_values=args.n_values,
        seeds=args.seeds,
        patch_path=args.patch,
        iterations=args.iterations,
        config=config,
    )

    # guardar datos crudos
    csv_path = output_dir / "scalability_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Datos guardados: {csv_path}")

    # generar figura
    plot_scalability(df, str(output_dir / "scalability_tcr_vs_n.png"))

    # resumen estadístico
    final = df[df["iteration"] == df["iteration"].max()]
    print("\n=== RESUMEN TCR FINAL ===")
    for n in sorted(final["n_drones"].unique()):
        vals = final.loc[final["n_drones"] == n, "tcr"]
        print(f"  N={n:2d}:  TCR = {vals.mean():.4f} ± {vals.std():.4f}  (min={vals.min():.4f}, max={vals.max():.4f})")


if __name__ == "__main__":
    main()
