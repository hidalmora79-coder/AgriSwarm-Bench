"""
Grid search enfocado: fija th=0.40, rho=0.05, barre alpha x step_size.
"""

import itertools
import json
import time
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment import AgriculturalField
from src.agent import Drone
from src.algorithms import HybridPSOACO
from src.metrics import update_coverage, calculate_tcr_from_mask, calculate_energy_stats
from src.config import get_profile, experiment_params, hybrid_params

PATCH_DIR = Path(__file__).resolve().parent.parent / "data" / "patches"
PATCHES = ["critico_1", "alto_7", "moderado_13", "saludable_19"]
SEEDS = 5
ITERS = 200

ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
STEPS = [3.0, 4.0, 5.0, 6.0, 8.0]


def make_field(patch_name, seed):
    field = AgriculturalField.from_ndvi_file(
        str(PATCH_DIR / f"{patch_name}.npy"), seed=seed
    )
    field.generate_ndvi_map()
    return field


def main():
    results = []
    total = len(ALPHAS) * len(STEPS)
    t0 = time.time()

    for i, (alpha, step) in enumerate(itertools.product(ALPHAS, STEPS)):
        params = {
            "hybrid.stress_threshold": 0.40,
            "hybrid.rho": 0.05,
            "hybrid.alpha": alpha,
            "hybrid.transit_step_size": step,
            "hybrid.beta": 2.0,
        }
        cfg = get_profile(params)
        hp = hybrid_params(cfg)
        exp = experiment_params(cfg)
        nd = exp["n_drones"]
        fs = exp["field_size"]
        soff = exp["start_offset_hybrid"]

        tcrs, energies = [], []

        for patch_name in PATCHES:
            for s in range(SEEDS):
                field = make_field(patch_name, s)
                rng = np.random.default_rng(s)
                drngs = rng.spawn(nd)

                targets = HybridPSOACO.get_niche_targets(
                    field.ndvi_map, nd, threshold=0.40
                )
                drones = []
                for di in range(nd):
                    offset = rng.uniform(-soff, soff, size=2)
                    start_pos = np.clip(targets[di] + offset, 0, fs - 1)
                    drones.append(Drone(di, start_pos, drngs[di]))

                optimizer = HybridPSOACO(
                    drones, fs, field.ndvi_map, config=cfg
                )
                cm = np.zeros((fs, fs), dtype=bool)

                for t in range(ITERS):
                    optimizer.step(drones, field)
                    update_coverage(cm, [d.pos for d in drones])
                    tcr = calculate_tcr_from_mask(field.ndvi_map, cm)
                    _, se = calculate_energy_stats(drones)

                tcrs.append(tcr)
                energies.append(se)

        entry = {
            "alpha": alpha,
            "step": step,
            "tcr_mean": float(np.mean(tcrs)),
            "tcr_std": float(np.std(tcrs)),
            "energy_mean": float(np.mean(energies)),
            "energy_std": float(np.std(energies)),
        }
        results.append(entry)

        eta = (time.time() - t0) / (i + 1) * (total - i - 1)
        print(
            f"[{i+1}/{total}] alpha={alpha:.2f} step={step:.1f} | "
            f"TCR={entry['tcr_mean']:.4f}+-{entry['tcr_std']:.4f} "
            f"E={entry['energy_mean']:.1f}+-{entry['energy_std']:.1f} | "
            f"ETA={eta:.0f}s"
        )

    results.sort(key=lambda r: r["tcr_mean"])
    out_path = (
        Path(__file__).resolve().parent.parent / "data" / "grid_search_focused.json"
    )
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== TOP 5 TCR ===")
    for r in results[:5]:
        print(
            f"  alpha={r['alpha']:.2f} step={r['step']:.1f} | "
            f"TCR={r['tcr_mean']:.4f}+-{r['tcr_std']:.4f} "
            f"E={r['energy_mean']:.1f}+-{r['energy_std']:.1f}"
        )

    print("\n=== TOP 5 ENERGY ===")
    by_e = sorted(results, key=lambda r: r["energy_mean"])
    for r in by_e[:5]:
        print(
            f"  alpha={r['alpha']:.2f} step={r['step']:.1f} | "
            f"TCR={r['tcr_mean']:.4f}+-{r['tcr_std']:.4f} "
            f"E={r['energy_mean']:.1f}+-{r['energy_std']:.1f}"
        )

    print(f"\nTiempo total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
