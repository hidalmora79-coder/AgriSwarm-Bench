import matplotlib.pyplot as plt
import numpy as np
import os
from src.environment import AgriculturalField
from src.agent import Drone
from src.algorithms import HybridPSOACO

def main():
    seed = 0
    config = 'HYBRID'
    iterations = 300
    
    # 1. Setup Entorno
    field = AgriculturalField(size=100, seed=seed)
    ndvi_map = field.generate_ndvi_map()
    
    main_rng = np.random.default_rng(seed)
    drone_rngs = main_rng.spawn(10)
    
    # 2. Inicialización de Drones con Despliegue Inteligente Unificado
    drones = []
    targets = HybridPSOACO.get_niche_targets(ndvi_map, 10)
    for i in range(10):
        offset = main_rng.uniform(-15, 15, size=2)
        start_pos = np.clip(targets[i] + offset, 0, 99)
        drones.append(Drone(i, start_pos, drone_rngs[i]))
        
    # 3. Inicialización Algoritmo Híbrido con Nichos
    algo = HybridPSOACO(drones, field.size, ndvi_map=ndvi_map, gamma=0.1)
    
    # 4. Ejecución de la Simulación
    print(f"Ejecutando simulación para visualización ({iterations} iteraciones)...")
    for t in range(iterations):
        algo.step(drones, field)
        
    # 5. Generación del Plot
    plt.figure(figsize=(12, 10))
    plt.imshow(ndvi_map, cmap='RdYlGn', vmin=0.2, vmax=0.9, alpha=0.7, origin='lower')
    plt.colorbar(label='NDVI Value')
    
    for drone in drones:
        path = np.array(drone.path)
        plt.plot(path[:, 0], path[:, 1], '-', linewidth=1.5, alpha=0.8, label=f'Drone {drone.id}')
        plt.scatter(drone.pos[0], drone.pos[1], s=100, marker='x', zorder=5)
        
    plt.title(f'Trayectorias - {config} (Seed {seed}) - Smart Deployment (Unified)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    plt.grid(alpha=0.3)
    
    output_path = 'trajectories_hybrid_s0_unified.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot guardado en {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
