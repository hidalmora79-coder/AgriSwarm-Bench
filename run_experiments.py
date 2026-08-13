"""
run_experiments.py

Orquestador principal de la simulación. Soporta múltiples escenarios
(sintéticos y reales), flotas heterogéneas, config YAML + CLI.
Exporta resultados legacy (Fase 1) y genera visualización de trayectorias.
"""

import numpy as np
import pandas as pd
import os
import csv
import argparse
from datetime import datetime
from pathlib import Path
from src.environment import AgriculturalField
from src.agent import Drone, ROLE_PROFILES
from src.algorithms import PurePSO, PureACO, HybridPSOACO, BoustrophedonCoverage, PSONiche
from src.metrics import update_coverage, calculate_tcr_from_mask, calculate_energy_stats
from src.config import (get_profile, experiment_params, build_overrides_from_args,
                        pso_params, aco_params, hybrid_params)

SCENARIO_SYNTHETIC = "synthetic"
SCENARIO_PATCH_DIR = Path("data/patches")


def discover_synthetic() -> list[dict]:
    return [{"name": SCENARIO_SYNTHETIC, "path": None}]


def discover_patches() -> list[dict]:
    if not SCENARIO_PATCH_DIR.exists():
        return []
    patches = sorted(SCENARIO_PATCH_DIR.glob("*.npy"))
    return [{"name": p.stem, "path": str(p)} for p in patches]


def get_scenarios(scenario_filter: str) -> list[dict]:
    all_s = discover_synthetic() + discover_patches()
    if scenario_filter == "all":
        return all_s
    names = [s.strip() for s in scenario_filter.split(",")]
    return [s for s in all_s if s["name"] in names]


def make_field(scenario: dict, seed: int, config: dict = None) -> AgriculturalField:
    exp = experiment_params(config)
    size = exp.get("field_size", 100)
    if scenario["path"] is None:
        field = AgriculturalField(size=size, seed=seed)
        field.generate_ndvi_map()
    else:
        field = AgriculturalField.from_ndvi_file(scenario["path"], seed=seed)
        field.generate_ndvi_map()
    return field


def build_fleet(n_drones: int, config_type: str, field,
                main_rng, drone_rngs, config: dict) -> list:
    """
    Construye flota heterogénea (monitor + sprayer) para HYBRID,
    u homogénea (monitor) para PSO/ACO.
    """
    exp = experiment_params(config)
    field_size = exp.get("field_size", 100)
    start_offset = exp.get("start_offset_hybrid", 15.0)
    fleet_mix = exp.get("fleet_mix", {"monitor": 6, "sprayer": 4})

    use_hybrid_fleet = config_type in ("HYBRID", "HYBRID_HETERO")
    if use_hybrid_fleet:
        hp = hybrid_params(config)
        targets = HybridPSOACO.get_niche_targets(
            field.ndvi_map, n_drones, threshold=hp.get("stress_threshold", 0.4)
        )

        if config_type == "HYBRID_HETERO":
            # Heterogeneous fleet: monitor + sprayer
            roles = (["monitor"] * fleet_mix.get("monitor", 6) +
                     ["sprayer"] * fleet_mix.get("sprayer", 4))
            roles = roles[:n_drones]
            while len(roles) < n_drones:
                roles.append("monitor")
            main_rng.shuffle(roles)
        else:
            # HYBRID: homogeneous all-monitor fleet for fair comparison
            roles = ["monitor"] * n_drones

        drones = []
        for i in range(n_drones):
            offset = main_rng.uniform(-start_offset, start_offset, size=2)
            start_pos = np.clip(targets[i] + offset, 0, field_size - 1)
            drones.append(Drone(i, start_pos, drone_rngs[i], role=roles[i]))
    else:
        # PSO/ACO: flota homogénea de monitores
        drones = []
        for i in range(n_drones):
            start_pos = (main_rng.uniform(0, field_size), main_rng.uniform(0, field_size))
            drones.append(Drone(i, start_pos, drone_rngs[i], role="monitor"))

    return drones


def run_single_simulation(seed: int, config_type: str, scenario: dict,
                          iterations: int = 300, config: dict = None):
    field = make_field(scenario, seed, config)
    exp = experiment_params(config)
    n_drones = exp.get("n_drones", 10)

    main_rng = np.random.default_rng(seed)
    drone_rngs = main_rng.spawn(n_drones)

    drones = build_fleet(n_drones, config_type, field, main_rng, drone_rngs, config)

    if config_type == "PSO":
        optimizer = PurePSO(drones, config=config)
    elif config_type == "ACO":
        optimizer = PureACO(field_size=field.size, config=config)
    elif config_type == "HYBRID" or config_type == "HYBRID_HETERO":
        optimizer = HybridPSOACO(drones, field_size=field.size, ndvi_map=field.ndvi_map,
                                 config=config)
    elif config_type == "BOUSTROPHEDON":
        optimizer = BoustrophedonCoverage(drones, field_size=field.size, config=config)
    elif config_type == "PSO_MULTI_NICHE":
        optimizer = PSONiche(drones, field_size=field.size, ndvi_map=field.ndvi_map,
                             config=config)
    else:
        raise ValueError(f"Unknown config: {config_type}")

    history = []
    covered_mask = np.zeros((field.size, field.size), dtype=bool)

    print(f"  > {config_type} (Seed {seed}) — {scenario['name']}")
    for t in range(iterations):
        optimizer.step(drones, field)

        current_positions = [d.pos for d in drones]
        update_coverage(covered_mask, current_positions)

        tcr = calculate_tcr_from_mask(field.ndvi_map, covered_mask)
        mean_e, std_e = calculate_energy_stats(drones)

        # Transit vs exploitation tracking
        stress_threshold = getattr(optimizer, "stress_threshold", 0.4)
        n_transit = 0
        n_exploit = 0
        for d in drones:
            if d.is_operational():
                dx, dy = int(round(d.pos[0])), int(round(d.pos[1]))
                dx = np.clip(dx, 0, field.size - 1)
                dy = np.clip(dy, 0, field.size - 1)
                if field.ndvi_map[dy, dx] < stress_threshold:
                    n_exploit += 1
                else:
                    n_transit += 1

        history.append({
            "seed": seed,
            "config": config_type,
            "scenario": scenario["name"],
            "iteration": t,
            "tcr": tcr,
            "mean_energy": mean_e,
            "sigma_energy": std_e,
            "n_transit": n_transit,
            "n_exploit": n_exploit,
        })

    return history, drones


def main():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo experiments with YAML config + CLI overrides"
    )
    parser.add_argument("--scenarios", default="synthetic",
                        help="Comma-separated scenarios or 'all' (default: synthetic)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of seeds per config (default: from config)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Iterations per run (default: from config)")
    parser.add_argument("--output", default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file (default: config/algorithms.yaml)")
    parser.add_argument("--param", action="append", default=[],
                        help="Override parameter: --param hybrid.alpha=2.0 (can be repeated)")
    parser.add_argument("--configs", default=None,
                        help="Comma-separated algorithm configs: PSO,ACO,HYBRID (default: all)")
    args = parser.parse_args()

    overrides = build_overrides_from_args(args.param)
    config = get_profile(overrides, args.config)
    exp = experiment_params(config)

    seeds = args.seeds or exp.get("default_seeds", 30)
    iterations = args.iterations or exp.get("default_iterations", 300)

    configurations = [c.strip() for c in args.configs.split(",")] if args.configs else ["PSO", "ACO", "HYBRID"]
    scenarios = get_scenarios(args.scenarios)

    if not scenarios:
        print(f"No scenarios matched filter: {args.scenarios}")
        print(f"Available: synthetic,{','.join(s['name'] for s in discover_patches())}")
        return

    output_dir = Path(args.output)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== EXPERIMENTO MONTE CARLO ===")
    print(f"Escenarios: {[s['name'] for s in scenarios]}")
    print(f"Configuraciones: {configurations}")
    print(f"Semillas: {seeds} | Iteraciones: {iterations}")
    print(f"Total corridas: {len(scenarios) * len(configurations) * seeds}")
    if args.param:
        print(f"Sobrescrituras: {overrides}")
    start_time = datetime.now()

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"ESCENARIO: {scenario['name']}")
        print(f"{'='*60}")
        sc_dir = raw_dir / scenario["name"]
        sc_dir.mkdir(parents=True, exist_ok=True)

        for cfg in configurations:
            print(f"\n> Configuración: {cfg}")
            for seed in range(seeds):
                results, drones = run_single_simulation(seed, cfg, scenario, iterations, config)
                df = pd.DataFrame(results)
                out_file = sc_dir / f"{cfg.lower()}_seed_{seed}.csv"
                df.to_csv(out_file, index=False)

    duration = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO FINALIZADO EN {duration.total_seconds():.2f}s")
    total_corridas = len(scenarios) * len(configurations) * seeds
    print(f"Corridas: {total_corridas}")
    print(f"Resultados: {raw_dir}/")

    # Export legacy CSV (Fase 1) para el primer escenario, HYBRID, seed=0
    if scenarios and "HYBRID" in configurations:
        first = scenarios[0]
        legacy_results, legacy_drones = run_single_simulation(
            0, "HYBRID", first, min(iterations, 50), config
        )
        legacy_path = Path("resultados_pso_aco_fase1.csv")
        with open(legacy_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Iteracion", "Tiempo_Cobertura_s", "Energia_Total_Wh", "TCR", "Robustez"])
            for row in legacy_results:
                w.writerow([
                    row["iteration"],
                    row["iteration"] * 0.5,
                    row["mean_energy"] * len(legacy_drones),
                    row["tcr"],
                    row["tcr"] * (sum(d.energy > 20 for d in legacy_drones) / len(legacy_drones)),
                ])
        print(f"Legacy CSV: {legacy_path.resolve()}")

        # Generar figura de trayectorias
        _plot_trajectories(first, legacy_drones, legacy_results)

    print(f"\n{'='*60}")


def _plot_trajectories(scenario: dict, drones: list, results: list):
    """Genera trayectorias_pso_aco.png (Fase 1 legacy)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Mapa NDVI
        field = make_field(scenario, 0)
        ndvi = field.ndvi_map
        axes[0].imshow(ndvi, cmap="RdYlGn", vmin=0, vmax=1)
        axes[0].set_title("Mapa NDVI", fontweight="bold")
        axes[0].set_axis_off()

        # Mapa de calor de cobertura
        cmap = np.zeros((field.size, field.size))
        for d in drones:
            for px, py in d.path:
                ix, iy = int(round(px)), int(round(py))
                if 0 <= ix < field.size and 0 <= iy < field.size:
                    cmap[iy, ix] += 1
        axes[1].imshow(cmap, cmap="hot")
        axes[1].set_title("Cobertura (frecuencia de visita)", fontweight="bold")
        axes[1].set_axis_off()

        # Trayectorias por rol
        colors = {"monitor": "#3498DB", "sprayer": "#E74C3C"}
        for d in drones:
            if d.path:
                xs, ys = zip(*d.path)
                axes[2].plot(xs, ys, color=colors.get(d.role, "#888"),
                             label=f"{d.role} {d.id}", alpha=0.7, linewidth=0.8)
        axes[2].set_title("Trayectorias - Flota Heterogénea", fontweight="bold")
        axes[2].legend(fontsize=6, ncol=2)
        axes[2].set_aspect("equal")
        axes[2].set_xlim(0, field.size)
        axes[2].set_ylim(0, field.size)

        plt.tight_layout()
        out = "trayectorias_pso_aco.png"
        fig.savefig(out, dpi=300)
        print(f"Trayectorias: {Path(out).resolve()}")
        plt.close(fig)
    except ImportError:
        print("  [WARN] matplotlib no disponible, saltando trayectorias")


if __name__ == "__main__":
    main()
