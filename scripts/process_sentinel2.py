"""
scripts/process_sentinel2.py

Procesa imágenes Sentinel-2 para generar parches NDVI 100×100 con
muestreo estratificado por nivel de estrés vegetal.

Flujo:
  1. Carga bandas B04 (rojo) y B08 (NIR) desde archivos .jp2 o .tif
  2. Calcula NDVI = (NIR - RED) / (NIR + RED + epsilon)
  3. Estratifica el mapa NDVI en bins de estrés
  4. Muestrea parches balanceados por estrato
  5. Guarda cada parche como .npy en data/patches/

Uso:
  python scripts/process_sentinel2.py --red B04.jp2 --nir B08.jp2
  python scripts/process_sentinel2.py --red B04.tif --nir B08.tif --n-patches 20 --strategy balanced
"""

import argparse
import numpy as np
from pathlib import Path


def compute_ndvi(red_path: str, nir_path: str) -> np.ndarray:
    import rasterio
    with rasterio.open(red_path) as red_src:
        red = red_src.read(1).astype(np.float64)
        profile = red_src.profile
    with rasterio.open(nir_path) as nir_src:
        nir = nir_src.read(1).astype(np.float64)

    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi, profile


def _extract_valid_patch(ndvi: np.ndarray, cy: int, cx: int,
                          patch_size: int) -> tuple | None:
    """Extrae un parche si cabe completamente en el mapa."""
    rows, cols = ndvi.shape
    half = patch_size // 2
    r_start = cy - half
    r_end = r_start + patch_size
    c_start = cx - half
    c_end = c_start + patch_size

    if r_start < 0 or r_end > rows or c_start < 0 or c_end > cols:
        return None
    patch = ndvi[r_start:r_end, c_start:c_end]
    if np.any(np.isnan(patch)):
        return None
    return r_start, c_start, patch


def extract_patches_stratified(ndvi: np.ndarray, patch_size: int = 100,
                                n_patches: int = 20) -> list[dict]:
    """
    Muestreo estratificado: divide el NDVI en bins y extrae parches
    balanceados por categoría de estrés.

    Estratos:
      - crítico:  NDVI < 0.2  (suelo desnudo / agua)
      - alto:     0.2 ≤ NDVI < 0.4  (estrés severo)
      - moderado: 0.4 ≤ NDVI < 0.6  (estrés ligero)
      - saludable: NDVI ≥ 0.6  (vegetación vigorosa)

    Returns:
        Lista de dicts con nombre descriptivo, coordenadas y patch NDVI
    """
    if n_patches < 4:
        n_patches = 4

    bins = [
        ("critico",  -1.0, 0.2),
        ("alto",      0.2, 0.4),
        ("moderado",  0.4, 0.6),
        ("saludable", 0.6, 1.0),
    ]

    rng = np.random.default_rng(42)
    patches = []
    selected = set()
    per_bin = max(1, n_patches // len(bins))

    for label, lo, hi in bins:
        mask = (ndvi >= lo) & (ndvi < hi)
        candidates = np.argwhere(mask)
        if len(candidates) == 0:
            continue

        attempts = 0
        max_attempts = per_bin * 50
        while len([p for p in patches if label in p["name"]]) < per_bin:
            if attempts >= max_attempts:
                break
            attempts += 1

            idx = rng.integers(len(candidates))
            cy, cx = candidates[idx]
            result = _extract_valid_patch(ndvi, cy, cx, patch_size)
            if result is None:
                continue
            r_start, c_start, patch = result
            key = (r_start, c_start)
            if key in selected:
                continue
            selected.add(key)

            veg_frac = np.sum(patch > 0.2) / (patch_size * patch_size)
            patch_id = len(patches) + 1
            patches.append({
                "name": f"{label}_{patch_id}",
                "row_start": r_start,
                "col_start": c_start,
                "ndvi_patch": patch.copy(),
                "veg_fraction": float(veg_frac),
                "mean_ndvi": float(np.mean(patch)),
                "min_ndvi": float(np.min(patch)),
                "stratum": label,
            })

    # Si faltan parches, completa con muestreo aleatorio global
    if len(patches) < n_patches:
        remaining = n_patches - len(patches)
        all_candidates = np.argwhere(np.ones_like(ndvi) if np.sum(np.isnan(ndvi)) > 0 else ~np.isnan(ndvi))
        attempts = 0
        while len([p for p in patches if "extra" in p["name"] or len(patches) < n_patches]) < remaining:
            if attempts > remaining * 100:
                break
            attempts += 1
            idx = rng.integers(len(all_candidates))
            cy, cx = all_candidates[idx]
            result = _extract_valid_patch(ndvi, cy, cx, patch_size)
            if result is None:
                continue
            r_start, c_start, patch = result
            key = (r_start, c_start)
            if key in selected:
                continue
            selected.add(key)
            patch_id = len(patches) + 1
            patches.append({
                "name": f"extra_{patch_id}",
                "row_start": r_start,
                "col_start": c_start,
                "ndvi_patch": patch.copy(),
                "veg_fraction": float(np.sum(patch > 0.2) / (patch_size * patch_size)),
                "mean_ndvi": float(np.mean(patch)),
                "min_ndvi": float(np.min(patch)),
                "stratum": "extra",
            })

    return patches


def extract_patches(ndvi: np.ndarray, patch_size: int = 100,
                    n_patches: int = 5, strategy: str = "veg") -> list[dict]:
    """
    Punto de entrada unificado para extracción de parches.

    Args:
        strategy: 'veg' → muestreo aleatorio en vegetación (original)
                  'balanced' → muestreo estratificado por nivel de estrés
    """
    rows, cols = ndvi.shape
    if rows < patch_size or cols < patch_size:
        raise ValueError(f"NDVI map ({rows}×{cols}) is smaller than patch size ({patch_size}×{patch_size})")

    if strategy == "balanced":
        return extract_patches_stratified(ndvi, patch_size, n_patches)

    # --- Estrategia original: vegetación ---
    veg_mask = ndvi > 0.2
    candidates = np.argwhere(veg_mask)
    if len(candidates) == 0:
        candidates = np.argwhere(np.ones_like(ndvi))

    rng = np.random.default_rng(42)
    selected = set()
    patches = []

    for i in range(n_patches * 3):
        if len(patches) >= n_patches:
            break
        idx = rng.integers(len(candidates))
        cy, cx = candidates[idx]
        result = _extract_valid_patch(ndvi, cy, cx, patch_size)
        if result is None:
            continue
        r_start, c_start, patch = result
        key = (r_start, c_start)
        if key in selected:
            continue
        selected.add(key)

        veg_frac = np.sum(patch > 0.2) / (patch_size * patch_size)
        patches.append({
            "name": f"patch_{len(patches) + 1}",
            "row_start": r_start,
            "col_start": c_start,
            "ndvi_patch": patch.copy(),
            "veg_fraction": float(veg_frac),
            "mean_ndvi": float(np.mean(patch)),
            "min_ndvi": float(np.min(patch)),
            "stratum": "veg",
        })

    return patches


def save_patches(patches: list[dict], output_dir: Path) -> list[Path]:
    """Guarda parches como .npy y metadatos extendidos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    info_lines = [
        "# name,row_start,col_start,veg_fraction,mean_ndvi,min_ndvi,stratum,file"
    ]
    for p in patches:
        fname = f"{p['name']}.npy"
        fpath = output_dir / fname
        np.save(str(fpath), p["ndvi_patch"])
        saved.append(fpath)
        info_lines.append(
            f"{p['name']},{p['row_start']},{p['col_start']},"
            f"{p['veg_fraction']:.4f},{p['mean_ndvi']:.4f},{p['min_ndvi']:.4f},"
            f"{p.get('stratum','')},{fname}"
        )

    log = output_dir / "patches_metadata.csv"
    log.write_text("\n".join(info_lines), encoding="utf-8")
    print(f"Metadata saved: {log}")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract stratified NDVI patches from Sentinel-2"
    )
    parser.add_argument("--red", required=True,
                        help="Path to Sentinel-2 B04 (Red) band .jp2 or .tif")
    parser.add_argument("--nir", required=True,
                        help="Path to Sentinel-2 B08 (NIR) band .jp2 or .tif")
    parser.add_argument("--output", default="data/patches",
                        help="Output directory for patches")
    parser.add_argument("--patch-size", type=int, default=100,
                        help="Patch size in pixels (default: 100)")
    parser.add_argument("--n-patches", type=int, default=20,
                        help="Number of patches to extract (default: 20)")
    parser.add_argument("--strategy", default="balanced",
                        choices=["veg", "balanced"],
                        help="Sampling strategy: 'veg' (random veg) or 'balanced' (stratified)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    print(f"Loading Red band: {args.red}")
    print(f"Loading NIR band: {args.nir}")
    ndvi, profile = compute_ndvi(args.red, args.nir)
    print(f"NDVI map shape: {ndvi.shape} | CRS: {profile.get('crs', 'N/A')}")
    print(f"  NDVI range: [{ndvi.min():.4f}, {ndvi.max():.4f}] | NaN: {np.isnan(ndvi).sum()}")

    print(f"Extracting {args.n_patches} patches ({args.strategy})...")
    patches = extract_patches(ndvi, patch_size=args.patch_size,
                               n_patches=args.n_patches, strategy=args.strategy)
    print(f"  Found {len(patches)} patches")

    if patches:
        print(f"\n  Resumen por estrato:")
        strata = set(p.get("stratum", "?") for p in patches)
        for s in sorted(strata):
            count = sum(1 for p in patches if p.get("stratum") == s)
            means = [p["mean_ndvi"] for p in patches if p.get("stratum") == s]
            print(f"    {s}: {count} parches | NDVI medio: {np.mean(means):.3f}")

    output_dir = Path(args.output)
    saved = save_patches(patches, output_dir)
    print(f"\nSaved {len(saved)} patches to {output_dir.resolve()}")
    for f in saved:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
