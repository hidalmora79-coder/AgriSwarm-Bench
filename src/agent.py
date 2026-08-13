"""
src/agent.py

Módulo que define la clase Drone con soporte para flotas heterogéneas
(monitor/sprayer), cinemática individual con velocidad por rol,
y modelo de consumo energético escalado.
"""

import numpy as np

ROLE_PROFILES = {
    "monitor": {
        "battery_capacity": 222.0,
        "max_velocity": 4.0,
        "energy_cost_per_unit": 0.10,
    },
    "sprayer": {
        "battery_capacity": 222.0,
        "max_velocity": 2.5,
        "energy_cost_per_unit": 0.15,
    },
}


class Drone:
    def __init__(self, drone_id: int, start_pos: tuple, rng: np.random.Generator,
                 role: str = "monitor"):
        profile = ROLE_PROFILES[role]
        self.id = drone_id
        self.role = role
        self.pos = np.array(start_pos, dtype=float)
        self.energy = profile["battery_capacity"]
        self.battery_capacity = profile["battery_capacity"]
        self.max_velocity = profile["max_velocity"]
        self.energy_cost = profile["energy_cost_per_unit"]
        self.rng = rng
        self.active = True
        self.path = [tuple(self.pos)]

    @property
    def residual_capacity_ratio(self) -> float:
        """Fracción de batería restante (0-1) para ponderación ACO."""
        return self.energy / self.battery_capacity if self.battery_capacity > 0 else 0.0

    def move_to(self, target_pos: tuple) -> float:
        if not self.active or self.energy <= 0:
            return 0.0
        target = np.array(target_pos, dtype=float)
        dx, dy = target - self.pos
        dist = np.linalg.norm([dx, dy])
        if dist < 1e-8:
            return 0.0
        step = min(self.max_velocity, dist)
        direction = np.array([dx / dist, dy / dist])
        new_pos = self.pos + direction * step
        consumed = step * self.energy_cost
        if self.energy >= consumed:
            self.pos = new_pos
            self.energy -= consumed
        else:
            fraction = self.energy / consumed
            self.pos = self.pos + direction * step * fraction
            self.energy = 0.0
            self.active = False
        self.path.append(tuple(self.pos))
        return consumed

    def is_operational(self) -> bool:
        return self.active and self.energy > 0
