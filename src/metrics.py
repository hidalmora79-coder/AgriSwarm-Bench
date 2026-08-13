"""
src/metrics.py

Módulo que implementa las funciones para el cálculo de indicadores de desempeño (KPIs)
de la simulación, incluyendo TCR y métricas estadísticas de energía.
"""

import numpy as np
from typing import List

def update_coverage(covered_mask: np.ndarray, new_positions: list, 
                    coverage_radius: float = 3.0) -> np.ndarray:
    """
    Actualiza la matriz de cobertura de forma incremental usando solo las nuevas posiciones.
    """
    rows, cols = covered_mask.shape
    r_int = int(np.ceil(coverage_radius))
    
    y_coords, x_coords = np.ogrid[:rows, :cols]

    for px, py in new_positions:
        ix, iy = int(round(px)), int(round(py))
        
        # Bounding box local
        x_min, x_max = max(0, ix - r_int), min(cols, ix + r_int + 1)
        y_min, y_max = max(0, iy - r_int), min(rows, iy + r_int + 1)
        
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (xx - px)**2 + (yy - py)**2
        
        # Operación OR in-place sobre la máscara persistente
        covered_mask[y_min:y_max, x_min:x_max] |= (dist_sq <= coverage_radius**2)
    
    return covered_mask

def calculate_tcr_from_mask(ndvi_map: np.ndarray, covered_mask: np.ndarray, 
                            target_threshold: float = 0.4) -> float:
    """
    Calcula el TCR basado en una máscara de cobertura pre-calculada.
    """
    targets = ndvi_map < target_threshold
    num_targets = np.sum(targets)
    if num_targets == 0: return 1.0
    
    covered_targets = np.logical_and(covered_mask, targets)
    return float(np.sum(covered_targets) / num_targets)

def calculate_energy_stats(drones: list) -> tuple:
    """
    Calcula el promedio y la desviación estándar de la energía residual de la flota.

    Args:
        drones: Lista de objetos de la clase Drone.

    Returns:
        tuple: (Energía residual promedio, Desviación estándar sigma_Energy).
    """
    energies = [d.energy for d in drones]
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    
    return float(mean_energy), float(std_energy)
