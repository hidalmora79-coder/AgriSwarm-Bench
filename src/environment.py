"""
src/environment.py

Módulo responsable de la generación y gestión del entorno de simulación.
Soporta mapas NDVI tanto sintéticos como provenientes de datos reales (Sentinel-2).
"""

import numpy as np
from pathlib import Path


class AgriculturalField:
    def __init__(self, size: int = 100, ndvi_min: float = 0.2, ndvi_max: float = 0.9,
                 seed: int = None, ndvi_map: np.ndarray = None, scenario: str = "synthetic"):
        self.size = size
        self.ndvi_min = ndvi_min
        self.ndvi_max = ndvi_max
        self.rng = np.random.default_rng(seed)
        self.scenario = scenario

        if ndvi_map is not None:
            if ndvi_map.shape != (size, size):
                raise ValueError(f"ndvi_map shape {ndvi_map.shape} != ({size}, {size})")
            self.ndvi_map = ndvi_map
        else:
            self.ndvi_map = None

    def generate_ndvi_map(self) -> np.ndarray:
        if self.ndvi_map is not None:
            return self.ndvi_map
        ndvi = np.full((self.size, self.size), 0.75)
        ndvi[20:40, 20:40] = self.rng.normal(0.30, 0.05, (20, 20))
        ndvi[60:85, 60:85] = self.rng.normal(0.35, 0.05, (25, 25))
        ndvi[15:30, 70:90] = self.rng.normal(0.25, 0.04, (15, 20))
        ndvi += self.rng.normal(0, 0.02, ndvi.shape)
        self.ndvi_map = np.clip(ndvi, self.ndvi_min, self.ndvi_max)
        return self.ndvi_map

    @classmethod
    def from_ndvi_file(cls, npy_path: str, seed: int = None, **kwargs) -> "AgriculturalField":
        ndvi = np.load(npy_path)
        size = ndvi.shape[0]
        assert ndvi.shape[1] == size, f"NDVI map must be square, got {ndvi.shape}"
        scenario = Path(npy_path).stem
        return cls(size=size, seed=seed, ndvi_map=ndvi, scenario=scenario, **kwargs)

    def get_stress_mask(self, threshold: float = 0.4) -> np.ndarray:
        return self.ndvi_map < threshold

    def get_stress_fraction(self, threshold: float = 0.4) -> float:
        return float(np.mean(self.get_stress_mask(threshold)))
