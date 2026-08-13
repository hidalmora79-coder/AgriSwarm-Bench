import matplotlib.pyplot as plt
from src.environment import AgriculturalField
import os

def main():
    # Create an instance of the field
    # Using a fixed seed for reproducibility in this visualization
    field = AgriculturalField(size=100, seed=42)
    ndvi_map = field.generate_ndvi_map()

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(ndvi_map, cmap='RdYlGn', vmin=0.2, vmax=0.9)
    plt.colorbar(label='NDVI Value')
    plt.title('Synthetic NDVI Map with Stochastic Stress Zones')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    # Save the plot
    output_path = 'ndvi_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
