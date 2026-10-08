import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_fault_properties_interactive(fault_table_path: str, fault_id: int, year_column: str):
    """
    Create an interactive 3D plot of fault properties using plotly.
    
    Parameters:
    -----------
    fault_table_path : str
        Path to the fault table CSV file
    fault_id : int
        The specific fault ID to plot
    year_column : str
        The year column name to plot (e.g., '2030', '2040', etc.)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive 3D plot
    """
    
    # Load the fault table
    fault_table = pd.read_csv(fault_table_path)
    
    # Filter data for the specific fault_id
    fault_data = fault_table[fault_table['fault_id'] == fault_id].copy()
    
    if fault_data.empty:
        raise ValueError(f"No data found for fault_id {fault_id}")
    
    # Check if the year column exists
    if year_column not in fault_data.columns:
        raise ValueError(f"Year column '{year_column}' not found in the fault table")
    
    # Check if spatial coordinate columns exist
    required_columns = ['x_ave', 'y_ave', 'z_ave']
    missing_columns = [col for col in required_columns if col not in fault_data.columns]
    if missing_columns:
        raise ValueError(f"Missing spatial coordinate columns: {missing_columns}")
    
    # Create interactive 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=fault_data['x_ave'],
        y=fault_data['y_ave'],
        z=fault_data['z_ave'],
        mode='markers',
        marker=dict(
            size=5,
            color=fault_data[year_column],
            colorscale='viridis',
            opacity=0.8,
            colorbar=dict(title=f'{year_column} Property Value')
        ),
        text=[f'X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}<br>Value: {val:.4f}' 
              for x, y, z, val in zip(fault_data['x_ave'], fault_data['y_ave'], 
                                     fault_data['z_ave'], fault_data[year_column])],
        hovertemplate='<b>Fault {}</b><br>'.format(fault_id) +
                     'X: %{x:.2f}<br>' +
                     'Y: %{y:.2f}<br>' +
                     'Z: %{z:.2f}<br>' +
                     'Property Value: %{marker.color:.4f}<extra></extra>'
    )])
    
    # Update layout for better 3D visualization
    fig.update_layout(
        title=f'Fault {fault_id} - {year_column} Properties (Interactive 3D View)',
        scene=dict(
            xaxis_title='X coordinate (m)',
            yaxis_title='Y coordinate (m)',
            zaxis_title='Z coordinate (m)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600
    )
    
    return fig

def plot_fault_properties_matplotlib(fault_table_path: str, fault_id: int, year_column: str):
    """
    Plot the properties of a specific fault across x_ave, y_ave, z_ave coordinates for a given year.
    This is the original matplotlib version for comparison.
    
    Parameters:
    -----------
    fault_table_path : str
        Path to the fault table CSV file
    fault_id : int
        The specific fault ID to plot
    year_column : str
        The year column name to plot (e.g., '2030', '2040', etc.)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot
    """
    
    # Load the fault table
    fault_table = pd.read_csv(fault_table_path)
    
    # Filter data for the specific fault_id
    fault_data = fault_table[fault_table['fault_id'] == fault_id].copy()
    
    if fault_data.empty:
        raise ValueError(f"No data found for fault_id {fault_id}")
    
    # Check if the year column exists
    if year_column not in fault_data.columns:
        raise ValueError(f"Year column '{year_column}' not found in the fault table")
    
    # Check if spatial coordinate columns exist
    required_columns = ['x_ave', 'y_ave', 'z_ave']
    missing_columns = [col for col in required_columns if col not in fault_data.columns]
    if missing_columns:
        raise ValueError(f"Missing spatial coordinate columns: {missing_columns}")
    
    # Create a single 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    
    # 3D scatter plot using spatial coordinates
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(fault_data['x_ave'], fault_data['y_ave'], fault_data['z_ave'], 
                         c=fault_data[year_column], cmap='viridis', s=50, alpha=0.7)
    
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.set_zlabel('Z coordinate (m)')
    ax.set_title(f'Fault {fault_id} - {year_column} Properties (3D Spatial View)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(f'{year_column} Property Value')
    
    plt.tight_layout()
    
    return fig

def plot_fault_timeseries(fault_table_path: str, fault_id: int, year_columns: list):
    """
    Plot the property values over time for a specific fault across all coordinates.
    
    Parameters:
    -----------
    fault_table_path : str
        Path to the fault table CSV file
    fault_id : int
        The specific fault ID to plot
    year_columns : list
        List of year column names to plot (e.g., ['2030', '2040', '2050'])
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot
    """
    
    # Load the fault table
    fault_table = pd.read_csv(fault_table_path)
    
    # Filter data for the specific fault_id
    fault_data = fault_table[fault_table['fault_id'] == fault_id].copy()
    
    if fault_data.empty:
        raise ValueError(f"No data found for fault_id {fault_id}")
    
    # Check if all year columns exist
    missing_columns = [col for col in year_columns if col not in fault_data.columns]
    if missing_columns:
        raise ValueError(f"Missing year columns: {missing_columns}")
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Property values over time for each coordinate
    ax1 = axes[0, 0]
    for idx, row in fault_data.iterrows():
        values = [row[year] for year in year_columns if pd.notna(row[year])]
        if values:  # Only plot if there are valid values
            ax1.plot(year_columns[:len(values)], values, 'o-', alpha=0.7, 
                    label=f"i={row['i']}, j={row['j']}, k={row['k']}")
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Property Value')
    ax1.set_title(f'Fault {fault_id} - Property Values Over Time')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot of property values by year
    ax2 = axes[0, 1]
    year_data = []
    for year in year_columns:
        year_values = fault_data[year].dropna()
        if not year_values.empty:
            year_data.append(year_values)
    
    if year_data:
        ax2.boxplot(year_data, labels=year_columns)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Property Value')
        ax2.set_title(f'Fault {fault_id} - Property Distribution by Year')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of property values across coordinates
    ax3 = axes[1, 0]
    # Create pivot table for heatmap
    pivot_data = fault_data.pivot_table(
        values=year_columns[0],  # Use first year for heatmap
        index='k', 
        columns='i', 
        aggfunc='mean'
    )
    
    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis', ax=ax3)
        ax3.set_title(f'Fault {fault_id} - {year_columns[0]} Property Heatmap (i vs k)')
        ax3.set_xlabel('i coordinate')
        ax3.set_ylabel('k coordinate')
    
    # Plot 4: Statistical summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    stats_text = f"Fault {fault_id} Statistics:\n\n"
    stats_text += f"Total coordinates: {len(fault_data)}\n"
    stats_text += f"i range: {fault_data['i'].min()} - {fault_data['i'].max()}\n"
    stats_text += f"j range: {fault_data['j'].min()} - {fault_data['j'].max()}\n"
    stats_text += f"k range: {fault_data['k'].min()} - {fault_data['k'].max()}\n\n"
    
    for year in year_columns:
        if year in fault_data.columns:
            year_values = fault_data[year].dropna()
            if not year_values.empty:
                stats_text += f"{year}:\n"
                stats_text += f"  Mean: {year_values.mean():.4f}\n"
                stats_text += f"  Std: {year_values.std():.4f}\n"
                stats_text += f"  Min: {year_values.min():.4f}\n"
                stats_text += f"  Max: {year_values.max():.4f}\n\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    return fig

def plot_fault_distribution_2d_matplotlib(fault_table_path: str, k_layer: int, year_column: str = None, figsize: tuple = (10, 8)):
    """
    Plot fault distribution across x_ave, y_ave coordinates for a specific k layer using matplotlib.
    
    Parameters:
    -----------
    fault_table_path : str
        Path to the fault table CSV file
    k_layer : int
        The specific k layer to plot
    year_column : str, optional
        If provided, color points by property value for that year
        If None, just show fault_id distribution
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (10, 8)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The 2D plot
    """
    
    # Load the fault table
    fault_table = pd.read_csv(fault_table_path)
    
    # Filter data for the specific k layer
    layer_data = fault_table[fault_table['k'] == k_layer].copy()
    
    if layer_data.empty:
        raise ValueError(f"No data found for k layer {k_layer}")
    
    # Check if spatial coordinate columns exist
    required_columns = ['x_ave', 'y_ave']
    missing_columns = [col for col in required_columns if col not in layer_data.columns]
    if missing_columns:
        raise ValueError(f"Missing spatial coordinate columns: {missing_columns}")
    
    # Create the plot with specified figure size
    fig, ax = plt.subplots(figsize=figsize)
    
    if year_column and year_column in layer_data.columns:
        # Color by property value
        scatter = ax.scatter(layer_data['x_ave'], layer_data['y_ave'], 
                           c=layer_data[year_column], cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label=f'{year_column} Property Value')
        title = f'Fault Distribution at k={k_layer} - {year_column} Properties'
    else:
        # Color by fault_id
        unique_faults = layer_data['fault_id'].unique()
        for fault_id in unique_faults:
            fault_data = layer_data[layer_data['fault_id'] == fault_id]
            ax.scatter(fault_data['x_ave'], fault_data['y_ave'], 
                      label=f'Fault {fault_id}', s=50, alpha=0.7)
        ax.legend()
        title = f'Fault Distribution at k={k_layer}'
    
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Example usage functions
def example_usage():
    """
    Example of how to use the plotting functions
    """
    # Example 1: Plot single year properties
    fig1 = plot_fault_properties_interactive(
        fault_table_path='results/JD_Sula_2025_flow_fault.csv',
        fault_id=1,  # Replace with actual fault_id
        year_column='2030'
    )
    fig1.show()
    
    # Example 2: Plot timeseries
    fig2 = plot_fault_timeseries(
        fault_table_path='results/JD_Sula_2025_flow_fault.csv',
        fault_id=1,  # Replace with actual fault_id
        year_columns=['2030', '2040', '2050', '2060']
    )
    plt.show()

if __name__ == "__main__":
    # You can run this directly or import the functions
    print("Fault plotting functions loaded. Use plot_fault_properties() or plot_fault_timeseries()") 