"""
agricultural_metrics.py

Fase 4 — Métricas Agrícolas @CEA
Post-procesamiento de datos Monte Carlo existentes para calcular:
  4A: Área monitoreada efectiva (ha/misión)
  4B: Cobertura por unidad de energía (ha/Wh)
  4C: Primera iteración de detección por zona de estrés
  4D: Cobertura parcial por umbral de tiempo (TCR en iter 50,100,150,200,300)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.environment import AgriculturalField
from src.metrics import update_coverage

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

CELL_SIZE_HA = 0.01  # 10m x 10m = 100 m2 = 0.01 ha
FIELD_SIZE = 100
TOTAL_AREA_HA = FIELD_SIZE * FIELD_SIZE * CELL_SIZE_HA  # 100 ha

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/agricultural_metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_COLORS = {"PSO": "#1f77b4", "ACO": "#ff7f0e", "HYBRID": "#2ca02c"}
CONFIG_LABELS = {"PSO": "Pure PSO", "ACO": "Pure ACO", "HYBRID": "Hybrid PSO-ACO"}

TCR_ITERATIONS = [49, 99, 149, 199, 299]  # 0-indexed: steps 50, 100, 150, 200, 300


def get_stressed_area_ha(seed: int = 0) -> float:
    """Compute stressed area (NDVI < 0.4) in hectares for the synthetic scenario."""
    field = AgriculturalField(size=FIELD_SIZE, seed=seed)
    field.generate_ndvi_map()
    stress_mask = field.get_stress_mask(threshold=0.4)
    stressed_cells = np.sum(stress_mask)
    return float(stressed_cells * CELL_SIZE_HA), stressed_cells, field.ndvi_map


def load_synthetic_data() -> pd.DataFrame:
    """Load all synthetic scenario CSVs."""
    syn_dir = RAW_DIR / "synthetic"
    if not syn_dir.exists():
        print("ERROR: synthetic data not found. Run experiments first.")
        sys.exit(1)
    frames = []
    for csv_file in sorted(syn_dir.glob("*.csv")):
        frames.append(pd.read_csv(csv_file))
    return pd.concat(frames, ignore_index=True)


def compute_agricultural_metrics(df: pd.DataFrame, stressed_ha: float):
    """Compute 4A (effective area) and 4B (efficiency) from existing data."""
    df = df.copy()
    df["area_ha"] = df["tcr"] * stressed_ha
    initial_energy = 222.0
    df["energy_consumed"] = initial_energy - df["mean_energy"]
    df["efficiency_ha_per_wh"] = np.where(
        df["energy_consumed"] > 0,
        df["area_ha"] / df["energy_consumed"],
        0.0
    )
    return df


def compute_partial_tcr(df: pd.DataFrame):
    """4D: Extract TCR at iterations 50, 100, 150, 200, 300."""
    return df[df["iteration"].isin(TCR_ITERATIONS)].copy()


def compute_zone_coverage_progression(num_seeds: int = 10, iterations: int = 300):
    """4C: Coverage progression per stress zone.
    Tracks zone-level coverage over time to measure how quickly
    each zone is fully inspected.
    """
    zones = [(20, 40, 20, 40), (60, 85, 60, 85), (15, 30, 70, 90)]
    zone_names = ["Zone 1 (core 20×20)", "Zone 2 (sparse 25×25)", "Zone 3 (strip 15×20)"]

    for seed in range(num_seeds):
        field = AgriculturalField(size=FIELD_SIZE, seed=seed)
        field.generate_ndvi_map()
        ndvi = field.ndvi_map

        from src.agent import Drone
        from src.algorithms import HybridPSOACO
        from src.config import get_profile, hybrid_params, experiment_params

        config = get_profile()
        hp = hybrid_params(config)
        exp = experiment_params(config)
        n_drones = exp.get("n_drones", 10)
        start_offset = exp.get("start_offset_hybrid", 15.0)

        targets = HybridPSOACO.get_niche_targets(ndvi, n_drones, threshold=hp.get("stress_threshold", 0.4))

        main_rng = np.random.default_rng(seed)
        drone_rngs = main_rng.spawn(n_drones)

        drones = []
        for i in range(n_drones):
            offset = main_rng.uniform(-start_offset, start_offset, size=2)
            start_pos = np.clip(targets[i] + offset, 0, FIELD_SIZE - 1)
            drones.append(Drone(i, start_pos, drone_rngs[i], role="monitor"))

        optimizer = HybridPSOACO(drones, field_size=FIELD_SIZE, ndvi_map=ndvi, config=config)
        zone_size = [(ymax - ymin) * (xmax - xmin) for (ymin, ymax, xmin, xmax) in zones]
        zone_covered = np.zeros(len(zones), dtype=np.int64)
        first_visit_recorded = np.full(len(zones), False)

        all_records = {zi: [] for zi in range(len(zones))}
        zone_visited = [np.zeros((zone_size[zi],), dtype=bool) for zi in range(len(zones))]

        for t in range(iterations):
            optimizer.step(drones, field)
            for d in drones:
                x, y = int(round(d.pos[0])), int(round(d.pos[1]))
                if 0 <= x < FIELD_SIZE and 0 <= y < FIELD_SIZE:
                    for zi, (ymin, ymax, xmin, xmax) in enumerate(zones):
                        if xmin <= x < xmax and ymin <= y < ymax:
                            if ndvi[y, x] < 0.4:
                                local_y = y - ymin
                                local_x = x - xmin
                                local_idx = local_y * (xmax - xmin) + local_x
                                if not zone_visited[zi][local_idx]:
                                    zone_visited[zi][local_idx] = True
                                    if not first_visit_recorded[zi]:
                                        first_visit_recorded[zi] = True

            # Record coverage every 10 iterations
            if t % 10 == 0 or t == iterations - 1:
                for zi in range(len(zones)):
                    zc = int(np.sum(zone_visited[zi]))
                    pct = zc / zone_size[zi] * 100
                    all_records[zi].append({"seed": seed, "zone": zi + 1, "zone_name": zone_names[zi],
                                            "iteration": t + 1, "cells_covered": zc,
                                            "zone_cells": zone_size[zi], "cover_pct": pct})

        if (seed + 1) % 5 == 0:
            print(f"  Zone tracking: seed {seed+1}/{num_seeds}")

    all_dfs = [pd.DataFrame(all_records[zi]) for zi in range(len(zones))]
    return pd.concat(all_dfs, ignore_index=True)


def plot_effective_area(df: pd.DataFrame, stressed_ha: float, output_path: str):
    """Figure: effective area (ha) comparison."""
    final = df[df["iteration"] == df["iteration"].max()].copy()
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = ["PSO", "ACO", "HYBRID"]
    labels = [CONFIG_LABELS[c] for c in configs]
    data = [final.loc[final["config"] == c, "area_ha"].values for c in configs]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", linewidth=1.8),
                     flierprops=dict(marker="o", markersize=4, alpha=0.5, color="gray"))

    for patch, color in zip(bp["boxes"], [CONFIG_COLORS[c] for c in configs]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    for w in bp["whiskers"]: w.set(color="black", linewidth=1.4)
    for c in bp["caps"]: c.set(color="black", linewidth=1.4)

    ax.set_ylabel("Effective Monitored Area (ha)")
    ax.set_title("Effective Coverage Area by Algorithm", pad=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.axhline(y=stressed_ha, color="red", linestyle="--", linewidth=1.2,
               label=f"Stressed area = {stressed_ha:.1f} ha")
    ax.legend()

    for i, (lbl, vals) in enumerate(zip(labels, data), start=1):
        m, s = np.mean(vals), np.std(vals)
        ax.text(i, 0.5, f"$\mu$={m:.2f}\n$\sigma$={s:.2f}",
                ha="center", va="bottom", fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.85))

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_efficiency(df: pd.DataFrame, output_path: str):
    """Figure: ha/Wh efficiency."""
    final = df[df["iteration"] == df["iteration"].max()].copy()
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = ["PSO", "ACO", "HYBRID"]
    labels = [CONFIG_LABELS[c] for c in configs]
    data = [final.loc[final["config"] == c, "efficiency_ha_per_wh"].values for c in configs]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", linewidth=1.8),
                     flierprops=dict(marker="o", markersize=4, alpha=0.5, color="gray"))

    for patch, color in zip(bp["boxes"], [CONFIG_COLORS[c] for c in configs]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    for w in bp["whiskers"]: w.set(color="black", linewidth=1.4)
    for c in bp["caps"]: c.set(color="black", linewidth=1.4)

    ax.set_ylabel("Efficiency (ha/Wh)")
    ax.set_title("Area Coverage per Unit Energy", pad=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    for i, (lbl, vals) in enumerate(zip(labels, data), start=1):
        m, s = np.mean(vals), np.std(vals)
        ax.text(i, max(vals) * 0.9, f"$\mu$={m:.6f}\n$\sigma$={s:.6f}",
                ha="center", va="top", fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.85))

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_partial_tcr(df_partial: pd.DataFrame, output_path: str):
    """Figure: TCR at iterations 50, 100, 150, 200, 300."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    configs = ["PSO", "ACO", "HYBRID"]
    x = np.arange(len(TCR_ITERATIONS))
    width = 0.25

    for ci, config in enumerate(configs):
        means = []
        stds = []
        for it in TCR_ITERATIONS:
            vals = df_partial.loc[(df_partial["config"] == config) & (df_partial["iteration"] == it), "tcr"]
            means.append(vals.mean())
            stds.append(vals.std())

        offset = (ci - 1) * width
        bars = ax.bar(x + offset + width / 2, means, width, yerr=stds,
                       label=CONFIG_LABELS[config], color=CONFIG_COLORS[config],
                       alpha=0.75, capsize=3, edgecolor="black", linewidth=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("TCR")
    ax.set_title("TCR Progression at Checkpoint Iterations", pad=10)
    ax.set_xticks(x + width / 2)
    display_iters = [it + 1 for it in TCR_ITERATIONS]
    ax.set_xticklabels([str(it) for it in display_iters])
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_zone_coverage(df_cover: pd.DataFrame, output_path: str):
    """Figure: zone coverage (%) over simulation time."""
    if df_cover.empty:
        print("  WARN: No zone coverage data to plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    zone_names = sorted(df_cover["zone_name"].unique())
    colors = ["#E74C3C", "#E67E22", "#3498DB"]

    for zi, zname in enumerate(zone_names):
        zdf = df_cover[df_cover["zone_name"] == zname]
        means = zdf.groupby("iteration")["cover_pct"].mean()
        stds = zdf.groupby("iteration")["cover_pct"].std()
        iters = means.index.values
        ax.plot(iters, means.values, color=colors[zi], linewidth=1.8, label=zname)
        ax.fill_between(iters, means.values - stds.values,
                         means.values + stds.values,
                         color=colors[zi], alpha=0.12)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Zone Coverage (%)")
    ax.set_title("Per-Zone Coverage Progression (Hybrid PSO-ACO)", pad=10)
    ax.legend(framealpha=0.9)
    ax.grid(linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Also print summary stats
    final = df_cover[df_cover["iteration"] == df_cover["iteration"].max()]
    print("\n  Final zone coverage at end of simulation:")
    for zi, zname in enumerate(zone_names):
        vals = final[final["zone_name"] == zname]["cover_pct"]
        print(f"    {zname}: {vals.mean():.1f}% +/- {vals.std():.1f}%")


def main():
    print("=== METRICAS AGRICOLAS (FASE 4) ===")
    print()

    # Stressed area for the synthetic scenario
    stressed_ha, stressed_cells, ndvi = get_stressed_area_ha(seed=0)
    print(f"Grid: {FIELD_SIZE}x{FIELD_SIZE} celdas = {TOTAL_AREA_HA:.0f} ha total")
    print(f"Celdas estresadas (NDVI < 0.4): {stressed_cells}")
    print(f"Area estresada: {stressed_ha:.2f} ha")
    print()

    # Load synthetic data
    df = load_synthetic_data()
    print(f"Datos cargados: {len(df)} registros, {df['config'].nunique()} configs, {df['seed'].nunique()} seeds")
    print()

    # 4A + 4B: Agricultural metrics
    final = df[df["iteration"] == df["iteration"].max()].copy()
    df_metrics = compute_agricultural_metrics(final, stressed_ha)

    print("--- 4A: AREA EFECTIVA (ha/mision) ---")
    for cfg in ["PSO", "ACO", "HYBRID"]:
        vals = df_metrics.loc[df_metrics["config"] == cfg, "area_ha"]
        print(f"  {cfg}: {vals.mean():.2f} +/- {vals.std():.2f} ha")

    print()
    print("--- 4B: EFICIENCIA (ha/Wh) ---")
    for cfg in ["PSO", "ACO", "HYBRID"]:
        vals = df_metrics.loc[df_metrics["config"] == cfg, "efficiency_ha_per_wh"]
        print(f"  {cfg}: {vals.mean():.6f} +/- {vals.std():.6f} ha/Wh")

    # 4D: TCR at specific iterations
    print()
    print("--- 4D: TCR POR ITERACION ---")
    df_partial = compute_partial_tcr(df)
    for cfg in ["PSO", "ACO", "HYBRID"]:
        print(f"  {cfg}:")
        for it in TCR_ITERATIONS:
            vals = df_partial.loc[(df_partial["config"] == cfg) & (df_partial["iteration"] == it), "tcr"]
            display_it = it + 1
            print(f"    iter {display_it:3d}: TCR = {vals.mean():.4f} +/- {vals.std():.4f}")

    # Figures
    print()
    print("--- GENERANDO FIGURAS ---")
    plot_effective_area(df_metrics, stressed_ha, str(OUTPUT_DIR / "effective_area.png"))
    plot_efficiency(df_metrics, str(OUTPUT_DIR / "efficiency_ha_per_wh.png"))
    plot_partial_tcr(df_partial, str(OUTPUT_DIR / "partial_tcr_progression.png"))

    # 4C: Zone coverage progression (requires re-running simulations)
    print()
    print("--- 4C: COBERTURA POR ZONA DE ESTRES ---")
    print("  Re-ejecutando simulacion con tracking por zona...")
    df_cover = compute_zone_coverage_progression(num_seeds=10, iterations=300)
    csv_cover = OUTPUT_DIR / "zone_coverage.csv"
    df_cover.to_csv(csv_cover, index=False)
    print(f"  Datos guardados: {csv_cover}")

    plot_zone_coverage(df_cover, str(OUTPUT_DIR / "zone_coverage_progression.png"))

    # Save combined metrics table
    print()
    print("--- TABLA RESUMEN ---")
    rows = []
    for cfg in ["PSO", "ACO", "HYBRID"]:
        f = df_metrics[df_metrics["config"] == cfg]
        rows.append({
            "Config": cfg,
            "TCR": f"{f['tcr'].mean():.3f} +/- {f['tcr'].std():.3f}",
            "Area (ha)": f"{f['area_ha'].mean():.2f} +/- {f['area_ha'].std():.2f}",
            "Eficiencia (ha/Wh)": f"{f['efficiency_ha_per_wh'].mean():.6f}",
            "Energy Residual (Wh)": f"{f['mean_energy'].mean():.1f} +/- {f['mean_energy'].std():.1f}",
        })
    summary = pd.DataFrame(rows)
    csv_summary = OUTPUT_DIR / "summary_metrics.csv"
    summary.to_csv(csv_summary, index=False)
    print(summary.to_string(index=False))
    print(f"\nTabla guardada: {csv_summary}")
    print(f"\nTodas las figuras: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
