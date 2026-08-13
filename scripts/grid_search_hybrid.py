"""
scripts/grid_search_hybrid.py

Grid search de hiperparámetros del algoritmo HybridPSOACO.
Barre combinaciones de stress_threshold, rho, beta, alpha y transit_step_size
sobre parches representativos de cada estrato.

Uso:
  uv run python scripts/grid_search_hybrid.py
  uv run python scripts/grid_search_hybrid.py --seeds 5 --patches saludable_19,alto_7,critico_1
  uv run python scripts/grid_search_hybrid.py --quick
"""

import argparse
import itertools
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment import AgriculturalField
from src.agent import Drone
from src.algorithms import HybridPSOACO
from src.metrics import update_coverage, calculate_tcr_from_mask, calculate_energy_stats
from src.config import get_profile, experiment_params, hybrid_params, build_overrides_from_args


PATCH_DIR = Path("data/patches")


def discover_patches_by_stratum() -> dict[str, list[str]]:
    if not PATCH_DIR.exists():
        return {}
    meta = PATCH_DIR / "patches_metadata.csv"
    if not meta.exists():
        return {}
    df = pd.read_csv(meta, skiprows=1,
                     names=["name","row_start","col_start","veg_fraction",
                            "mean_ndvi","min_ndvi","stratum","file"])
    strata = {}
    for _, row in df.iterrows():
        s = row.get("stratum", "extra")
        fname = row.get("file", "")
        strata.setdefault(s, []).append(fname.replace(".npy", ""))
    return strata


def make_field(patch_name: str, seed: int) -> AgriculturalField:
    npy_path = PATCH_DIR / f"{patch_name}.npy"
    field = AgriculturalField.from_ndvi_file(str(npy_path), seed=seed)
    field.generate_ndvi_map()
    return field


def evaluate_params(params: dict, patches: list[str],
                    seeds: int, iterations: int) -> dict:
    """Evalúa una combinación de parámetros y retorna métricas agregadas."""
    config = get_profile(params)
    exp = experiment_params(config)
    hp = hybrid_params(config)
    n_drones = exp.get("n_drones", 10)
    field_size = exp.get("field_size", 100)
    start_offset = exp.get("start_offset_hybrid", 15.0)

    tcr_values = []
    energy_values = []

    for patch_name in patches:
        for seed in range(seeds):
            field = make_field(patch_name, seed)
            main_rng = np.random.default_rng(seed)
            drone_rngs = main_rng.spawn(n_drones)

            targets = HybridPSOACO.get_niche_targets(
                field.ndvi_map, n_drones,
                threshold=hp.get("stress_threshold", 0.4)
            )

            drones = []
            for i in range(n_drones):
                offset = main_rng.uniform(-start_offset, start_offset, size=2)
                start_pos = np.clip(targets[i] + offset, 0, field_size - 1)
                drones.append(Drone(i, start_pos, drone_rngs[i]))

            optimizer = HybridPSOACO(drones, field_size=field.size,
                                     ndvi_map=field.ndvi_map, config=config)

            covered_mask = np.zeros((field.size, field.size), dtype=bool)
            final_tcr = 0.0
            final_sigma = 0.0

            for t in range(iterations):
                optimizer.step(drones, field)
                current_positions = [d.pos for d in drones]
                update_coverage(covered_mask, current_positions)
                final_tcr = calculate_tcr_from_mask(field.ndvi_map, covered_mask)
                _, final_sigma = calculate_energy_stats(drones)

            tcr_values.append(final_tcr)
            energy_values.append(final_sigma)

    return {
        "tcr_mean": float(np.mean(tcr_values)),
        "tcr_std": float(np.std(tcr_values)),
        "energy_mean": float(np.mean(energy_values)),
        "energy_std": float(np.std(energy_values)),
        "n_runs": len(tcr_values),
    }


def main():
    parser = argparse.ArgumentParser(description="Grid search for HYBRID PSO-ACO")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per param combo")
    parser.add_argument("--iterations", type=int, default=200, help="Iterations per run")
    parser.add_argument("--quick", action="store_true", help="Quick reduced grid")
    parser.add_argument("--patches", default=None,
                        help="Comma-separated patch names (default: one per stratum)")
    parser.add_argument("--output", default="data/grid_search_results.json")
    args = parser.parse_args()

    # Seleccionar parches representativos
    if args.patches:
        patches = [p.strip() for p in args.patches.split(",")]
    else:
        strata = discover_patches_by_stratum()
        patches = []
        for s in ["critico", "alto", "moderado", "saludable"]:
            if s in strata and strata[s]:
                patches.append(strata[s][0])
        if not patches:
            patches = ["patch_1", "patch_2", "patch_3"]
    print(f"Patches: {patches}")

    # --- Grid de hiperparámetros ---
    if args.quick:
        thresholds = [0.35, 0.40, 0.45]
        rhos = [0.05, 0.1, 0.2]
        betas = [1.0, 2.0, 3.0]
        alphas = [1.0]
        step_sizes = [5.0]
    else:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
        rhos = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
        betas = [0.5, 1.0, 2.0, 3.0, 4.0]
        alphas = [0.5, 1.0, 1.5, 2.0]
        step_sizes = [3.0, 5.0, 8.0]

    grid = list(itertools.product(thresholds, rhos, betas, alphas, step_sizes))
    total = len(grid)
    print(f"\nGrid: {total} combinaciones")
    print(f"  thresholds: {thresholds}")
    print(f"  rho:        {rhos}")
    print(f"  beta:       {betas}")
    print(f"  alpha:      {alphas}")
    print(f"  step_size:  {step_sizes}")

    results = []
    start = time.time()

    for i, (th, rho, beta, alpha, step) in enumerate(grid):
        params = {
            "hybrid.stress_threshold": th,
            "hybrid.rho": rho,
            "hybrid.beta": beta,
            "hybrid.alpha": alpha,
            "hybrid.transit_step_size": step,
        }

        metrics = evaluate_params(params, patches, args.seeds, args.iterations)

        entry = {
            "stress_threshold": th,
            "rho": rho,
            "beta": beta,
            "alpha": alpha,
            "transit_step_size": step,
            **metrics,
        }
        results.append(entry)

        elapsed = time.time() - start
        eta = (elapsed / (i + 1)) * (total - i - 1)
        print(
            f"  [{i+1}/{total}] th={th:.2f} rho={rho:.2f} beta={beta:.1f} "
            f"alpha={alpha:.1f} step={step:.1f} | "
            f"TCR={metrics['tcr_mean']:.4f}±{metrics['tcr_std']:.4f} "
            f"E={metrics['energy_mean']:.1f}±{metrics['energy_std']:.1f} "
            f"| ETA={eta:.0f}s"
        )

    # Ordenar por TCR (menor es mejor)
    results.sort(key=lambda r: r["tcr_mean"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"RESULTADOS ({len(results)} combos)")
    print(f"{'='*60}")
    print(f"\n-- Top 5 por TCR (menor mejor) --")
    for r in results[:5]:
        print(
            f"  th={r['stress_threshold']:.2f} rho={r['rho']:.2f} "
            f"beta={r['beta']:.1f} alpha={r['alpha']:.1f} step={r['transit_step_size']:.1f} | "
            f"TCR={r['tcr_mean']:.4f}±{r['tcr_std']:.4f} "
            f"E={r['energy_mean']:.1f}±{r['energy_std']:.1f}"
        )

    print(f"\n-- Top 5 por Sigma-Energía (menor mejor) --")
    by_energy = sorted(results, key=lambda r: r["energy_mean"])
    for r in by_energy[:5]:
        print(
            f"  th={r['stress_threshold']:.2f} rho={r['rho']:.2f} "
            f"beta={r['beta']:.1f} alpha={r['alpha']:.1f} step={r['transit_step_size']:.1f} | "
            f"TCR={r['tcr_mean']:.4f}±{r['tcr_std']:.4f} "
            f"E={r['energy_mean']:.1f}±{r['energy_std']:.1f}"
        )

    best = results[0]
    print(f"\n-- Mejor combinación global --")
    print(json.dumps(best, indent=2))
    print(f"\nResultados guardados: {output.resolve()}")
    print(f"Tiempo total: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
