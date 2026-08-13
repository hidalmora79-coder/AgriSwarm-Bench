"""
scripts/figures_publication.py

Genera figuras publicables (estilo IEEE/Elsevier) con los resultados
de los experimentos Monte Carlo con HYBRID PSO-ACO tuneado.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

RAW_DIR = Path("data/raw")
FIG_DIR = Path("data/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"PSO": "#E74C3C", "ACO": "#3498DB", "HYBRID": "#2ECC71"}
MARKERS = {"PSO": "s", "ACO": "o", "HYBRID": "D"}
LABELS = {"PSO": "PSO", "ACO": "ACO", "HYBRID": "Hybrid PSO-ACO"}


def load_results() -> pd.DataFrame:
    rows = []
    for scenario in sorted(RAW_DIR.iterdir()):
        if not scenario.is_dir():
            continue
        for csv in scenario.glob("*.csv"):
            parts = csv.stem.split("_seed_")
            config = parts[0].upper()
            seed = int(parts[1])
            df = pd.read_csv(csv)
            rows.append(df.assign(config=config, seed=seed))
    return pd.concat(rows, ignore_index=True)


def load_metadata():
    meta_path = Path("data/patches/patches_metadata.csv")
    if not meta_path.exists():
        return {}
    df = pd.read_csv(meta_path, skiprows=1,
                     names=["name", "row", "col", "veg_frac", "mean_ndvi",
                            "min_ndvi", "stratum", "file"])
    return dict(zip(df.name, df.stratum))


def fig_boxplot_tcr_by_stratum(full: pd.DataFrame, stratum_map: dict):
    """Boxplot TCR final agrupado por estrato y configuración."""
    final = full[full.iteration == full.iteration.max()].copy()
    final["stratum"] = final.scenario.map(stratum_map).fillna("synthetic")

    strata_order = ["critico", "alto", "moderado", "saludable"]
    configs = ["PSO", "ACO", "HYBRID"]

    fig, axes = plt.subplots(1, len(strata_order), figsize=(14, 4.5), sharey=True)

    for ax, stratum in zip(axes, strata_order):
        data = final[final.stratum == stratum]
        positions = []
        boxes = []
        for j, cfg in enumerate(configs):
            vals = data[data.config == cfg].tcr.values
            if len(vals) == 0:
                continue
            positions.append(j)
            boxes.append(vals)
            bp = ax.boxplot(vals, positions=[j], widths=0.5, patch_artist=True,
                            showfliers=False,
                            boxprops=dict(facecolor=COLORS[cfg], alpha=0.7),
                            medianprops=dict(color="black", linewidth=1.5),
                            whiskerprops=dict(color=COLORS[cfg]),
                            capprops=dict(color=COLORS[cfg]))

        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([LABELS[c] for c in configs], rotation=25)
        ax.set_title(f"{stratum.capitalize()}", fontweight="bold")
        ax.set_ylabel("TCR" if stratum == strata_order[0] else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.suptitle("Tasa de Cobertura de Rescate (TCR) — Final (t=300)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIG_DIR / "pub_boxplot_tcr_stratum.png"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def fig_convergence_tcr_by_stratum(full: pd.DataFrame, stratum_map: dict):
    """Curvas de convergencia TCR promedio por estrato."""
    full["stratum"] = full.scenario.map(stratum_map).fillna("synthetic")
    strata_order = ["critico", "alto", "moderado", "saludable"]
    configs = ["PSO", "ACO", "HYBRID"]

    fig, axes = plt.subplots(1, len(strata_order), figsize=(14, 4.5), sharey=True)

    for ax, stratum in zip(axes, strata_order):
        data = full[full.stratum == stratum]
        for cfg in configs:
            sub = data[data.config == cfg]
            grouped = sub.groupby("iteration").tcr.agg(["mean", "std"])
            iters = grouped.index.values
            mean = grouped["mean"].values
            std = grouped["std"].values
            ax.plot(iters, mean, color=COLORS[cfg], label=LABELS[cfg],
                    linewidth=1.5)
            ax.fill_between(iters, mean - std, mean + std,
                            color=COLORS[cfg], alpha=0.1)

        ax.set_title(f"{stratum.capitalize()}", fontweight="bold")
        ax.set_xlabel("Iteración")
        ax.set_ylabel("TCR" if stratum == strata_order[0] else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        if stratum == strata_order[-1]:
            ax.legend(framealpha=0.8, edgecolor="gray")

    fig.suptitle("Convergencia TCR — Promedio sobre 10 semillas",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIG_DIR / "pub_convergence_tcr_stratum.png"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def fig_energy_vs_tcr_scatter(full: pd.DataFrame, stratum_map: dict):
    """Scatter TCR vs Sigma-Energy (compromiso cobertura-eficiencia)."""
    final = full[full.iteration == full.iteration.max()].copy()
    final["stratum"] = final.scenario.map(stratum_map).fillna("synthetic")
    real = final[final.stratum != "synthetic"]
    configs = ["PSO", "ACO", "HYBRID"]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for cfg in configs:
        sub = real[real.config == cfg]
        ax.scatter(sub.tcr, sub.sigma_energy,
                   c=COLORS[cfg], label=LABELS[cfg], alpha=0.4, s=20,
                   marker=MARKERS[cfg])
        # centroide
        cx, cy = sub.tcr.mean(), sub.sigma_energy.mean()
        ax.scatter(cx, cy, c=COLORS[cfg], s=120, marker=MARKERS[cfg],
                   edgecolors="black", linewidths=1.5, zorder=5)

    ax.set_xlabel("TCR (menor = mejor cobertura)")
    ax.set_ylabel("Sigma-Energía (menor = más estable)")
    ax.legend(framealpha=0.8, edgecolor="gray")
    ax.set_title("Compromiso Cobertura vs Eficiencia Energética",
                 fontweight="bold")

    # Anotar centroides
    for cfg in configs:
        sub = real[real.config == cfg]
        ax.annotate(f"{LABELS[cfg]}",
                    (sub.tcr.mean(), sub.sigma_energy.mean()),
                    xytext=(5, 5), textcoords="offset points",
                    fontweight="bold", fontsize=10)

    plt.tight_layout()
    path = FIG_DIR / "pub_scatter_tcr_energy.png"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def fig_tcr_improvement_heatmap(full: pd.DataFrame, stratum_map: dict):
    """Mapa de calor: mejora relativa % de HYBRID sobre ACO y PSO por estrato."""
    final = full[full.iteration == full.iteration.max()].copy()
    final["stratum"] = final.scenario.map(stratum_map).fillna("synthetic")
    strata = ["critico", "alto", "moderado", "saludable"]
    configs = ["PSO", "ACO", "HYBRID"]

    means = final.groupby(["stratum", "config"]).tcr.mean().unstack()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    improvement_data = []
    for s in strata:
        hybrid_val = means.loc[s, "HYBRID"]
        imp_vs_pso = (means.loc[s, "PSO"] - hybrid_val) / means.loc[s, "PSO"] * 100
        imp_vs_aco = (means.loc[s, "ACO"] - hybrid_val) / means.loc[s, "ACO"] * 100
        improvement_data.append([imp_vs_pso, imp_vs_aco])

    im = ax.imshow(improvement_data, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=100)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["vs PSO", "vs ACO"])
    ax.set_yticks(range(len(strata)))
    ax.set_yticklabels([s.capitalize() for s in strata])

    for i in range(len(strata)):
        for j in range(2):
            val = improvement_data[i][j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontweight="bold", fontsize=11,
                    color="white" if val > 50 else "black")

    ax.set_title("Mejora TCR de HYBRID (%)", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Mejora (%)")
    plt.tight_layout()
    path = FIG_DIR / "pub_improvement_heatmap.png"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    print("=== FIGURAS PUBLICABLES ===")
    full = load_results()
    stratum_map = load_metadata()
    print(f"Total registros: {len(full):,}")
    print(f"Escenarios: {full.scenario.nunique()}")
    print(f"Semillas por config: {full.seed.nunique()}")

    fig_boxplot_tcr_by_stratum(full, stratum_map)
    fig_convergence_tcr_by_stratum(full, stratum_map)
    fig_energy_vs_tcr_scatter(full, stratum_map)
    fig_tcr_improvement_heatmap(full, stratum_map)

    print(f"\nFiguras guardadas en: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
