from custom_c_shaped_coil import CShapedCoil, verify_coil
import simnibs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_coil(coil):
    """
    Create 3D visualization of the coil design
    """
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get wire positions
    vertices = coil.coil_vertices
    
    # Plot the wire path
    ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'b-', linewidth=1, label='Wire Path')
    
    # Add scatter points for better visibility of wire positions
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r', s=1)
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('C-Shaped Coil Design')
    
    # Convert units to mm for display
    ax.set_xlim([-5e-3, 5e-3])
    ax.set_ylim([-5e-3, 5e-3])
    ax.set_zlim([0, 3e-3])
    
    # Add a grid
    ax.grid(True)
    
    # Add legend
    ax.legend()
    
    # Save the plot
    plot_dir = os.path.join(os.path.expanduser('~/tms_grid_project'), 'visualizations')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, 'coil_design_3d.png'), dpi=300, bbox_inches='tight')
    
    # Show different views
    views = [(0, 0), (90, 0), (0, 90)]  # azim, elev pairs
    for i, (azim, elev) in enumerate(views):
        ax.view_init(elev, azim)
        plt.savefig(os.path.join(plot_dir, f'coil_design_view_{i}.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_field(coil):
    """
    Create visualization of the magnetic field
    """
    # Create a grid of points for field calculation
    x = np.linspace(-5e-3, 5e-3, 50)
    y = np.linspace(-5e-3, 5e-3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Calculate field magnitude at each point (simplified)
    field_mag = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            r = np.sqrt((X[i,j])**2 + (Y[i,j])**2)
            field_mag[i,j] = coil.didt / (r + 1e-6)  # Simplified field calculation
    
    # Create figure for field visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Plot field magnitude
    im = ax.pcolormesh(X*1000, Y*1000, field_mag, shading='auto')
    plt.colorbar(im, label='Field Magnitude (A/m)')
    
    # Set labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Magnetic Field Magnitude in XY Plane (Z=0)')
    
    # Save the plot
    plot_dir = os.path.join(os.path.expanduser('~/tms_grid_project'), 'visualizations')
    plt.savefig(os.path.join(plot_dir, 'coil_field.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    try:
        # Create coil
        print("Creating coil design...")
        coil_designer = CShapedCoil()
        coil = coil_designer.create_coil_definition()
        
        # Verify coil
        print("\nVerifying coil properties...")
        if verify_coil(coil):
            print("Coil verification successful!")
        else:
            print("Coil verification failed!")
            exit(1)
        
        # Visualize coil
        print("\nCreating visualizations...")
        visualize_coil(coil)
        visualize_field(coil)
        
        print("\nVisualizations have been saved to the 'visualizations' directory.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)