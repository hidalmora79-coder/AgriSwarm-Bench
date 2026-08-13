"""run_scalability_synthetic.py

Re-run scalability analysis on synthetic scenario with 600 iters.
Only runs N=5, 20, 30 (N=10 already available from main campaign).
Saves to data/scalability_synthetic/.
"""
import subprocess, sys
from pathlib import Path

EXTRA_N = [5, 20, 30]
SEEDS = 30
ITERS = 600
WORKDIR = Path(__file__).parent.resolve()
OUTDIR = WORKDIR / "data" / "scalability_synthetic"
OUTDIR.mkdir(parents=True, exist_ok=True)

for n in EXTRA_N:
    print(f"\n{'='*60}")
    print(f"Running scalability N={n} on synthetic...")
    print(f"{'='*60}")
    
    result = subprocess.run([
        sys.executable, "run_experiments.py",
        "--configs", "HYBRID",
        "--iterations", str(ITERS),
        "--seeds", str(SEEDS),
        "--param", f"experiment.n_drones={n}",
        "--param", "experiment.fleet_mix.monitor=10",
        "--param", "experiment.fleet_mix.sprayer=0",
    ], cwd=WORKDIR, capture_output=True, text=True)
    print(result.stdout[-300:] if len(result.stdout) > 300 else result.stdout)
    if result.returncode != 0:
        print(f"ERROR (N={n}): {result.stderr[-300:]}")
        continue
    
    src = WORKDIR / "data" / "raw" / "synthetic"
    for f in sorted(src.glob("hybrid_seed_*.csv")):
        dest = OUTDIR / f"hybrid_n{n}_seed_{f.name.split('_seed_')[1]}"
        f.rename(dest)
        print(f"  -> {dest.name}")

# Copy N=10 from main data
src_main = WORKDIR / "data" / "raw" / "synthetic"
for f in sorted(src_main.glob("hybrid_seed_*.csv")):
    dest = OUTDIR / f"hybrid_n10_seed_{f.name.split('_seed_')[1]}"
    import shutil
    shutil.copy2(f, dest)
    print(f"  Copied {f.name} -> {dest.name}")

# Analyze
print(f"\nAnalyzing scalability results...")
import numpy as np
import pandas as pd

for n in [5, 10, 20, 30]:
    paths = sorted(OUTDIR.glob(f"hybrid_n{n}_seed_*.csv"))
    if not paths:
        continue
    dfs = [pd.read_csv(p) for p in paths]
    combined = pd.concat(dfs, ignore_index=True)
    final = combined[combined["iteration"] == combined["iteration"].max()]
    tcr = final["tcr"]
    print(f"  N={n:2d}: TCR={tcr.mean():.4f}+-{tcr.std():.4f}")

print(f"\nAll data in: {OUTDIR}/")
