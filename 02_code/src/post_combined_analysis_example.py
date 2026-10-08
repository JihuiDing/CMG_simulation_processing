#!/usr/bin/env python3
"""
Combined analysis example using both pressure data and grid coordinates.
This script demonstrates how to work with both datasets together.
"""

import numpy as np
import matplotlib.pyplot as plt

def load_both_datasets():
    """Load both pressure data and grid coordinates."""
    
    try:
        # Load pressure data
        pressure_data = np.load("results/pressure_data.npy")
        time_info = np.load("results/time_info.npy", allow_pickle=True).item()
        
        # Load grid coordinates
        grid_coordinates = np.load("results/grid_coordinates.npy")
        grid_info = np.load("results/grid_info.npy", allow_pickle=True).item()
        
        print("Both datasets loaded successfully!")
        print(f"Pressure data shape: {pressure_data.shape}")
        print(f"Grid coordinates shape: {grid_coordinates.shape}")
        
        # Verify dimensions match
        if pressure_data.shape[:3] == grid_coordinates.shape[:3]:
            print("✓ Grid dimensions match between pressure and coordinates")
        else:
            print("✗ Grid dimensions do not match!")
            return None, None, None, None
        
        return pressure_data, time_info, grid_coordinates, grid_info
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None, None, None

def example_1_pressure_at_coordinates(pressure_data, time_info, grid_coordinates, grid_info):
    """Example 1: Find pressure values at specific coordinate locations."""
    
    print("\n=== Example 1: Pressure at Specific Coordinates ===")
    
    # Define target coordinates (in meters)
    target_x = 450000.0  # X coordinate
    target_y = 7100000.0  # Y coordinate
    target_z = 1500.0     # Z coordinate
    
    print(f"Looking for pressure at coordinates: X={target_x}, Y={target_y}, Z={target_z}")
    
    # Find the closest grid cell
    x_coords = grid_coordinates[:,:,:,0]
    y_coords = grid_coordinates[:,:,:,1]
    z_coords = grid_coordinates[:,:,:,2]
    
    # Calculate distances
    distances = np.sqrt((x_coords - target_x)**2 + (y_coords - target_y)**2 + (z_coords - target_z)**2)
    
    # Find minimum distance
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    i, j, k = min_idx
    
    actual_coords = grid_coordinates[i, j, k, :]
    actual_distance = distances[i, j, k]
    
    print(f"Closest grid cell: I={i+1}, J={j+1}, K={k+1}")
    print(f"Actual coordinates: X={actual_coords[0]:.2f}, Y={actual_coords[1]:.2f}, Z={actual_coords[2]:.2f}")
    print(f"Distance: {actual_distance:.2f} m")
    
    # Get pressure history for this cell
    pressure_history = pressure_data[i, j, k, :]
    
    print(f"\nPressure history for this cell:")
    for t, (time, date) in enumerate(zip(time_info['time_values'], time_info['time_dates'])):
        print(f"  {date}: {pressure_history[t]:.2f} kPa")
    
    return i, j, k

def example_2_pressure_gradient_analysis(pressure_data, time_info, grid_coordinates, grid_info):
    """Example 2: Analyze pressure gradients using actual spatial coordinates."""
    
    print("\n=== Example 2: Pressure Gradient Analysis ===")
    
    # Select a specific time step and layer
    time_idx = -1  # Last time step
    k_layer = 39   # Middle layer
    
    pressure_slice = pressure_data[:, :, k_layer, time_idx]
    x_coords = grid_coordinates[:, :, k_layer, 0]
    y_coords = grid_coordinates[:, :, k_layer, 1]
    
    print(f"Analyzing pressure gradients for K={k_layer+1}, Time={time_info['time_dates'][time_idx]}")
    
    # Calculate pressure gradients using actual spatial coordinates
    # Use numpy.gradient for finite differences
    grad_x, grad_y = np.gradient(pressure_slice, x_coords[:, 0], y_coords[0, :])
    
    # Calculate gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    print(f"Pressure gradient statistics:")
    print(f"  Max X-gradient: {np.max(np.abs(grad_x)):.2f} kPa/m")
    print(f"  Max Y-gradient: {np.max(np.abs(grad_y)):.2f} kPa/m")
    print(f"  Max gradient magnitude: {np.max(grad_magnitude):.2f} kPa/m")
    
    # Find regions with highest gradients
    max_grad_idx = np.unravel_index(np.argmax(grad_magnitude), grad_magnitude.shape)
    max_grad_i, max_grad_j = max_grad_idx
    
    max_grad_coords = grid_coordinates[max_grad_i, max_grad_j, k_layer, :]
    max_grad_pressure = pressure_slice[max_grad_i, max_grad_j]
    
    print(f"\nHighest gradient location:")
    print(f"  I={max_grad_i+1}, J={max_grad_j+1}, K={k_layer+1}")
    print(f"  Coordinates: X={max_grad_coords[0]:.2f}, Y={max_grad_coords[1]:.2f}, Z={max_grad_coords[2]:.2f}")
    print(f"  Pressure: {max_grad_pressure:.2f} kPa")
    print(f"  Gradient magnitude: {grad_magnitude[max_grad_i, max_grad_j]:.2f} kPa/m")

def example_3_3d_pressure_visualization(pressure_data, time_info, grid_coordinates, grid_info):
    """Example 3: Create 3D visualization of pressure and coordinates."""
    
    print("\n=== Example 3: 3D Pressure Visualization ===")
    
    try:
        # Select a specific time step
        time_idx = -1  # Last time step
        pressure_3d = pressure_data[:, :, :, time_idx]
        
        # Get coordinates
        x_coords = grid_coordinates[:, :, :, 0]
        y_coords = grid_coordinates[:, :, :, 1]
        z_coords = grid_coordinates[:, :, :, 2]
        
        print(f"Creating 3D visualization for {time_info['time_dates'][time_idx]}")
        
        # Create a 3D scatter plot (sampling every few cells to avoid overcrowding)
        sample_rate = 5  # Sample every 5th cell
        i_sample = np.arange(0, pressure_3d.shape[0], sample_rate)
        j_sample = np.arange(0, pressure_3d.shape[1], sample_rate)
        k_sample = np.arange(0, pressure_3d.shape[2], sample_rate)
        
        x_sample = x_coords[i_sample][:, j_sample][:, :, k_sample]
        y_sample = y_coords[i_sample][:, j_sample][:, :, k_sample]
        z_sample = z_coords[i_sample][:, j_sample][:, :, k_sample]
        p_sample = pressure_3d[i_sample][:, j_sample][:, :, k_sample]
        
        # Flatten arrays for plotting
        x_flat = x_sample.flatten()
        y_flat = y_sample.flatten()
        z_flat = z_sample.flatten()
        p_flat = p_sample.flatten()
        
        # Filter out zero pressure values
        non_zero_mask = p_flat > 0
        x_plot = x_flat[non_zero_mask]
        y_plot = y_flat[non_zero_mask]
        z_plot = z_flat[non_zero_mask]
        p_plot = p_flat[non_zero_mask]
        
        print(f"Plotting {len(p_plot)} non-zero pressure cells")
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot with pressure as color
        scatter = ax.scatter(x_plot, y_plot, z_plot, c=p_plot, cmap='viridis', alpha=0.6, s=1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Pressure Distribution - {time_info["time_dates"][time_idx]}\n(Sampled every {sample_rate} cells)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Pressure (kPa)')
        
        plt.savefig(f'results/3d_pressure_distribution_t{time_idx}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("3D visualization saved to results/ directory")
        
    except Exception as e:
        print(f"Warning: Could not create 3D visualization: {e}")

def example_4_coordinate_based_analysis(pressure_data, time_info, grid_coordinates, grid_info):
    """Example 4: Coordinate-based pressure analysis."""
    
    print("\n=== Example 4: Coordinate-Based Analysis ===")
    
    # Define a region of interest using coordinates
    x_min, x_max = 440000, 460000
    y_min, y_max = 7090000, 7110000
    z_min, z_max = 1000, 2000
    
    print(f"Analyzing region:")
    print(f"  X: {x_min} to {x_max} m")
    print(f"  Y: {y_min} to {y_max} m")
    print(f"  Z: {z_min} to {z_max} m")
    
    # Find cells within this region
    x_coords = grid_coordinates[:,:,:,0]
    y_coords = grid_coordinates[:,:,:,1]
    z_coords = grid_coordinates[:,:,:,2]
    
    region_mask = ((x_coords >= x_min) & (x_coords <= x_max) &
                   (y_coords >= y_min) & (y_coords <= y_max) &
                   (z_coords >= z_min) & (z_coords <= z_max))
    
    region_cells = np.sum(region_mask)
    print(f"Cells in region: {region_cells}")
    
    if region_cells > 0:
        # Analyze pressure in this region for all time steps
        print(f"\nPressure analysis in region:")
        
        for t, (time, date) in enumerate(zip(time_info['time_values'], time_info['time_dates'])):
            pressure_slice = pressure_data[:,:,:,t]
            region_pressure = pressure_slice[region_mask]
            
            if len(region_pressure) > 0:
                non_zero_pressure = region_pressure[region_pressure > 0]
                if len(non_zero_pressure) > 0:
                    print(f"  {date}:")
                    print(f"    Mean pressure: {np.mean(non_zero_pressure):.2f} kPa")
                    print(f"    Max pressure: {np.max(non_zero_pressure):.2f} kPa")
                    print(f"    Min pressure: {np.min(non_zero_pressure):.2f} kPa")
                else:
                    print(f"  {date}: All pressures are zero")
            else:
                print(f"  {date}: No data")
    else:
        print("No cells found in specified region")

def main():
    """Main function to run combined analysis examples."""
    
    print("Combined Pressure and Grid Coordinates Analysis")
    print("=" * 55)
    
    # Load both datasets
    pressure_data, time_info, grid_coordinates, grid_info = load_both_datasets()
    
    if pressure_data is None:
        return 1
    
    print("\nDatasets loaded successfully!")
    
    # Run examples
    example_1_pressure_at_coordinates(pressure_data, time_info, grid_coordinates, grid_info)
    example_2_pressure_gradient_analysis(pressure_data, time_info, grid_coordinates, grid_info)
    example_3_3d_pressure_visualization(pressure_data, time_info, grid_coordinates, grid_info)
    example_4_coordinate_based_analysis(pressure_data, time_info, grid_coordinates, grid_info)
    
    print("\n" + "=" * 55)
    print("Combined analysis completed successfully!")
    print("Check the results/ directory for generated visualizations.")
    
    return 0

if __name__ == "__main__":
    exit(main()) 