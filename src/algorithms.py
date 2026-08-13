"""
src/algorithms.py

Módulo que implementa los algoritmos de optimización para la coordinación
de enjambres: PSO Puro, ACO Puro e Híbrido PSO-ACO con Nichos.
Soporta configuración por diccionario y por argumentos individuales.
"""

import numpy as np
from scipy.ndimage import label, center_of_mass
from src.config import pso_params, aco_params, hybrid_params


def _merge_config(params_cls, config: dict = None, **kwargs) -> dict:
    """Fusiona defaults de config YAML con kwargs explícitos (ignora None)."""
    p = params_cls(config)
    for k, v in kwargs.items():
        if v is not None:
            p[k] = v
    return {k: v for k, v in p.items() if not k.startswith("_")}


class PurePSO:
    def __init__(self, drones: list, config: dict = None,
                 w: float = None, c1: float = None, c2: float = None,
                 v_max: float = None):
        p = _merge_config(pso_params, config, w=w, c1=c1, c2=c2, v_max=v_max)
        self.w = p["w"]
        self.c1 = p["c1"]
        self.c2 = p["c2"]
        self.v_max = p["v_max"]
        self.pbest_pos = [np.copy(d.pos) for d in drones]
        self.pbest_fitness = [float('-inf')] * len(drones)
        self.gbest_pos = np.copy(drones[0].pos)
        self.gbest_fitness = float('-inf')
        self.velocities = [np.zeros(2) for _ in drones]

    def _calculate_fitness(self, pos: np.ndarray, ndvi_map: np.ndarray) -> float:
        rows, cols = ndvi_map.shape
        x, y = int(round(pos[0])), int(round(pos[1]))
        x, y = np.clip(x, 0, cols - 1), np.clip(y, 0, rows - 1)
        return float(1.0 - ndvi_map[y, x])

    def step(self, drones: list, field: object):
        ndvi_map = field.ndvi_map
        for i, drone in enumerate(drones):
            if not drone.is_operational(): continue
            current_fitness = self._calculate_fitness(drone.pos, ndvi_map)
            if current_fitness > self.pbest_fitness[i]:
                self.pbest_fitness[i] = current_fitness
                self.pbest_pos[i] = np.copy(drone.pos)
            if current_fitness > self.gbest_fitness:
                self.gbest_fitness = current_fitness
                self.gbest_pos = np.copy(drone.pos)

        for i, drone in enumerate(drones):
            if not drone.is_operational(): continue
            r1, r2 = drone.rng.random(), drone.rng.random()
            cognitive = self.c1 * r1 * (self.pbest_pos[i] - drone.pos)
            social = self.c2 * r2 * (self.gbest_pos - drone.pos) if self.gbest_fitness > float('-inf') else np.zeros(2)

            self.velocities[i] = self.w * self.velocities[i] + cognitive + social
            speed = np.linalg.norm(self.velocities[i])
            if speed > self.v_max:
                self.velocities[i] = (self.velocities[i] / speed) * self.v_max

            new_pos = np.clip(drone.pos + self.velocities[i], 0, field.size - 1)
            drone.move_to(tuple(new_pos))


class PureACO:
    def __init__(self, field_size: int, config: dict = None,
                 alpha: float = None, beta: float = None, rho: float = None,
                 Q: float = None, initial_pheromone: float = None,
                 pheromone_floor: float = None, micro_steps: int = None):
        p = _merge_config(aco_params, config, alpha=alpha, beta=beta, rho=rho,
                          Q=Q, initial_pheromone=initial_pheromone,
                          pheromone_floor=pheromone_floor, micro_steps=micro_steps)
        self.alpha = p["alpha"]
        self.beta = p["beta"]
        self.rho = p["rho"]
        self.Q = p["Q"]
        self.initial_pheromone = p["initial_pheromone"]
        self.pheromone_floor = p["pheromone_floor"]
        self.micro_steps = p["micro_steps"]
        self.pheromones = np.ones((field_size, field_size)) * self.initial_pheromone

    def step(self, drones: list, field: object):
        ndvi_map = field.ndvi_map
        rows, cols = ndvi_map.shape
        for drone in drones:
            if not drone.is_operational(): continue
            for _ in range(self.micro_steps):
                if not drone.is_operational(): break
                cx, cy = int(round(drone.pos[0])), int(round(drone.pos[1]))
                adj_coords, adj_probs = [], []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < cols and 0 <= ny < rows:
                            tau = self.pheromones[ny, nx] ** self.alpha
                            eta = (1.0 - ndvi_map[ny, nx]) ** self.beta
                            adj_coords.append((nx, ny))
                            adj_probs.append(tau * eta)
                if not adj_probs: break
                prob_sum = sum(adj_probs)
                adj_probs = [p / prob_sum for p in adj_probs] if prob_sum > 0 else [1.0 / len(adj_probs)] * len(adj_probs)
                target_idx = drone.rng.choice(len(adj_coords), p=adj_probs)
                drone.move_to(adj_coords[target_idx])
                fitness = 1.0 - ndvi_map[int(round(drone.pos[1])), int(round(drone.pos[0]))]
                self.pheromones[int(round(drone.pos[1])), int(round(drone.pos[0]))] += self.Q * fitness
        self.pheromones *= (1.0 - self.rho)
        self.pheromones = np.clip(self.pheromones, self.pheromone_floor, None)


class HybridPSOACO:
    def __init__(self, drones: list, field_size: int, ndvi_map: np.ndarray,
                 config: dict = None,
                 w: float = None, c1: float = None, c2: float = None,
                 alpha: float = None, beta: float = None, rho: float = None,
                 Q: float = None, stress_threshold: float = None,
                 transit_step_size: float = None, aco_micro_steps: int = None,
                 initial_pheromone: float = None, pheromone_floor: float = None):
        p = _merge_config(hybrid_params, config, w=w, c1=c1, c2=c2,
                          alpha=alpha, beta=beta, rho=rho, Q=Q,
                          stress_threshold=stress_threshold,
                          transit_step_size=transit_step_size,
                          aco_micro_steps=aco_micro_steps,
                          initial_pheromone=initial_pheromone,
                          pheromone_floor=pheromone_floor)
        self.stress_threshold = p["stress_threshold"]
        self.transit_step_size = p["transit_step_size"]
        self.aco_micro_steps = p["aco_micro_steps"]
        self.aco = PureACO(field_size, config=config, alpha=alpha, beta=beta,
                           rho=rho, Q=Q, initial_pheromone=initial_pheromone,
                           pheromone_floor=pheromone_floor,
                           micro_steps=aco_micro_steps)
        self.drone_targets = self.get_niche_targets(ndvi_map, len(drones),
                                                    threshold=self.stress_threshold)
        self.w = p["w"]
        self.c1 = p["c1"]
        self.c2 = p["c2"]
        self.v_max = p.get("v_max", self.transit_step_size)
        self.velocities = [np.zeros(2) for _ in drones]
        self.pbest_pos = [np.copy(d.pos) for d in drones]
        self.pbest_fitness = [float("-inf")] * len(drones)

    @staticmethod
    def get_niche_targets(ndvi_map: np.ndarray, num_drones: int,
                          threshold: float = 0.4) -> list:
        stress_mask = ndvi_map < threshold
        labeled_mask, num_features = label(stress_mask)
        if num_features > 0:
            centroids_raw = center_of_mass(stress_mask, labeled_mask, range(1, num_features + 1))
            centroids = [np.array([c[1], c[0]]) for c in centroids_raw]
            zone_sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
            total_area = sum(zone_sizes)
            drones_per_zone = [max(1, int(round((size / total_area) * num_drones))) for size in zone_sizes]
            while sum(drones_per_zone) < num_drones:
                drones_per_zone[np.argmax(zone_sizes)] += 1
            while sum(drones_per_zone) > num_drones:
                drones_per_zone[np.argmax(drones_per_zone)] -= 1
            targets = []
            for idx, count in enumerate(drones_per_zone):
                for _ in range(count):
                    targets.append(centroids[idx])
            return targets
        return [np.array([50.0, 50.0])] * num_drones

    def step(self, drones: list, field: object):
        ndvi_map = field.ndvi_map
        rows, cols = ndvi_map.shape

        for i, drone in enumerate(drones):
            if not drone.is_operational(): continue

            ix, iy = int(round(drone.pos[0])), int(round(drone.pos[1]))
            ix, iy = np.clip(ix, 0, cols - 1), np.clip(iy, 0, rows - 1)
            is_in_stress = ndvi_map[iy, ix] < self.stress_threshold

            if not is_in_stress:
                niche_target = self.drone_targets[i]
                vector = niche_target - drone.pos
                dist = np.linalg.norm(vector)

                if dist > 1.0:
                    # Canonical PSO transit mode with w, c1, c2, r1, r2
                    current_fitness = 1.0 - ndvi_map[iy, ix]
                    if current_fitness > self.pbest_fitness[i]:
                        self.pbest_fitness[i] = current_fitness
                        self.pbest_pos[i] = np.copy(drone.pos)

                    r1, r2 = drone.rng.random(), drone.rng.random()
                    cognitive = self.c1 * r1 * (self.pbest_pos[i] - drone.pos)
                    social = self.c2 * r2 * (niche_target - drone.pos)

                    v_max = getattr(drone, "max_velocity", self.v_max)
                    self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                    speed = np.linalg.norm(self.velocities[i])
                    if speed > v_max:
                        self.velocities[i] = (self.velocities[i] / speed) * v_max

                    new_pos = np.clip(drone.pos + self.velocities[i], 0, field.size - 1)
                    drone.move_to(tuple(new_pos))
                else:
                    is_in_stress = True

            if is_in_stress:
                for _ in range(self.aco_micro_steps):
                    if not drone.is_operational(): break

                    cx, cy = int(round(drone.pos[0])), int(round(drone.pos[1]))
                    adj_coords, adj_probs = [], []

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0: continue
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < cols and 0 <= ny < rows:
                                tau = self.aco.pheromones[ny, nx] ** self.aco.alpha
                                eta = (1.0 - ndvi_map[ny, nx]) ** self.aco.beta
                                adj_coords.append((nx, ny))
                                adj_probs.append(tau * eta)

                    if not adj_probs: break
                    prob_sum = sum(adj_probs)
                    adj_probs = [p / prob_sum for p in adj_probs] if prob_sum > 0 else [1.0 / len(adj_probs)] * len(adj_probs)

                    target_idx = drone.rng.choice(len(adj_coords), p=adj_probs)
                    drone.move_to(adj_coords[target_idx])

                    # Depósito de feromona ponderado por capacidad residual (Eq. 4.3)
                    fitness = 1.0 - ndvi_map[int(round(drone.pos[1])), int(round(drone.pos[0]))]
                    res_w = getattr(drone, "residual_capacity_ratio", 1.0)
                    self.aco.pheromones[int(round(drone.pos[1])), int(round(drone.pos[0]))] += self.aco.Q * fitness * res_w

        self.aco.pheromones *= (1.0 - self.aco.rho)
        self.aco.pheromones = np.clip(self.aco.pheromones, self.aco.pheromone_floor, None)


class BoustrophedonCoverage:
    """
    Deterministic lawnmower coverage baseline.

    Assigns each drone a horizontal band of the field.
    Sweeps left-right across the band; upon encountering a stress cell
    (NDVI < threshold), performs a dense local micro-scan.
    Used by commercial ag drones (real-world baseline).
    """

    def __init__(self, drones: list, field_size: int, config: dict = None,
                 stress_threshold: float = None, sweep_step: float = None,
                 descent_step: float = None, micro_steps: int = None):
        from src.config import boustrophedon_params
        p = boustrophedon_params(config)
        self.stress_threshold = stress_threshold if stress_threshold is not None else p["stress_threshold"]
        self.sweep_step = sweep_step if sweep_step is not None else p["sweep_step"]
        self.descent_step = descent_step if descent_step is not None else p["descent_step"]
        self.micro_steps = micro_steps if micro_steps is not None else p["micro_steps"]
        self.field_size = field_size

        n = len(drones)
        band_height = field_size / n
        self.drone_states = []
        for i in range(n):
            y_center = i * band_height + band_height / 2
            self.drone_states.append({"going_right": True})
            drones[i].pos = np.array([0.0, y_center])

    def step(self, drones: list, field: object):
        ndvi_map = field.ndvi_map
        for i, drone in enumerate(drones):
            if not drone.is_operational():
                continue
            state = self.drone_states[i]
            ix = int(round(drone.pos[0]))
            iy = int(round(drone.pos[1]))
            ix = np.clip(ix, 0, self.field_size - 1)
            iy = np.clip(iy, 0, self.field_size - 1)

            if ndvi_map[iy, ix] < self.stress_threshold:
                for _ in range(self.micro_steps):
                    if not drone.is_operational():
                        break
                    direction = 1.0 if state["going_right"] else -1.0
                    nx = np.clip(drone.pos[0] + direction * 1.0, 0, self.field_size - 1)
                    drone.move_to((nx, drone.pos[1]))
            else:
                direction = 1.0 if state["going_right"] else -1.0
                nx = drone.pos[0] + direction * self.sweep_step
                if nx < 0:
                    nx = 0.0
                    state["going_right"] = True
                    ny = np.clip(drone.pos[1] + self.descent_step, 0, self.field_size - 1)
                    drone.move_to((nx, ny))
                elif nx >= self.field_size:
                    nx = self.field_size - 1
                    state["going_right"] = False
                    ny = np.clip(drone.pos[1] + self.descent_step, 0, self.field_size - 1)
                    drone.move_to((nx, ny))
                else:
                    drone.move_to((nx, drone.pos[1]))


class PSONiche:
    """
    PSO with per-drone niche targets (no ACO/pheromones).

    Allocates drones to stress zones using the same niche assignment
    as HybridPSOACO. Drones use PSO to navigate toward and search
    within their assigned zone. Isolates the tactical ACO contribution
    of the hybrid algorithm.
    """

    def __init__(self, drones: list, field_size: int, ndvi_map: np.ndarray,
                 config: dict = None,
                 w: float = None, c1: float = None, c2: float = None,
                 v_max: float = None, stress_threshold: float = None,
                 transit_step_size: float = None):
        p = _merge_config(pso_params, config, w=w, c1=c1, c2=c2, v_max=v_max)
        self.w = p["w"]
        self.c1 = p["c1"]
        self.c2 = p["c2"]
        self.v_max = p["v_max"]
        self.stress_threshold = stress_threshold if stress_threshold is not None else 0.4
        self.transit_step_size = transit_step_size if transit_step_size is not None else 5.0

        self.drone_targets = HybridPSOACO.get_niche_targets(
            ndvi_map, len(drones), threshold=self.stress_threshold
        )
        self.pbest_pos = [np.copy(d.pos) for d in drones]
        self.pbest_fitness = [float("-inf")] * len(drones)
        self.gbest_pos = np.copy(drones[0].pos)
        self.gbest_fitness = float("-inf")
        self.velocities = [np.zeros(2) for _ in drones]

    def _calculate_fitness(self, pos: np.ndarray, ndvi_map: np.ndarray) -> float:
        rows, cols = ndvi_map.shape
        x, y = int(round(pos[0])), int(round(pos[1]))
        x, y = np.clip(x, 0, cols - 1), np.clip(y, 0, rows - 1)
        return float(1.0 - ndvi_map[y, x])

    def step(self, drones: list, field: object):
        ndvi_map = field.ndvi_map
        for i, drone in enumerate(drones):
            if not drone.is_operational():
                continue
            cf = self._calculate_fitness(drone.pos, ndvi_map)
            if cf > self.pbest_fitness[i]:
                self.pbest_fitness[i] = cf
                self.pbest_pos[i] = np.copy(drone.pos)
            if cf > self.gbest_fitness:
                self.gbest_fitness = cf
                self.gbest_pos = np.copy(drone.pos)

            ix = int(round(drone.pos[0]))
            iy = int(round(drone.pos[1]))
            ix = np.clip(ix, 0, field.size - 1)
            iy = np.clip(iy, 0, field.size - 1)
            in_stress = ndvi_map[iy, ix] < self.stress_threshold

            if not in_stress:
                target = self.drone_targets[i]
                vector = target - drone.pos
                dist = np.linalg.norm(vector)
                if dist > 1.0:
                    v_max = getattr(drone, "max_velocity", self.transit_step_size)
                    step = min(dist, v_max)
                    move = (vector / dist) * step
                    np_pos = np.clip(drone.pos + move, 0, field.size - 1)
                    drone.move_to(tuple(np_pos))
                    continue

            r1, r2 = drone.rng.random(), drone.rng.random()
            cognitive = self.c1 * r1 * (self.pbest_pos[i] - drone.pos)
            social = self.c2 * r2 * (self.gbest_pos - drone.pos)
            self.velocities[i] = self.w * self.velocities[i] + cognitive + social
            spd = np.linalg.norm(self.velocities[i])
            if spd > self.v_max:
                self.velocities[i] = (self.velocities[i] / spd) * self.v_max
            np_pos = np.clip(drone.pos + self.velocities[i], 0, field.size - 1)
            drone.move_to(tuple(np_pos))
