import simnibs
import numpy as np
import os

class CShapedCoil:
    def __init__(self, 
                 length=7e-3,      # 7 mm
                 width=4e-3,       # 4 mm
                 height=3e-3,      # 3 mm
                 gap_width=0.5e-3, # 0.5 mm
                 n_turns=30,
                 wire_diameter=0.1e-3):  # 0.1 mm
        
        self.length = length
        self.width = width
        self.height = height
        self.gap_width = gap_width
        self.n_turns = n_turns
        self.wire_diameter = wire_diameter
        
    def create_coil_definition(self):
        """Create the C-shaped coil definition"""
        # Create a TMS coil list structure
        coil = simnibs.sim_struct.TMSLIST()
        
        # Set position and direction (required attributes in TMSLIST)
        coil.fnamecoil = 'custom_c_shaped'  # Name of the coil
        coil.pos = [0, 0, 0]  # Position in space
        coil.matsimnibs = np.eye(4)  # Transformation matrix
        coil.didt = 1e6  # Rate of current change (A/s)
        
        # Add the wire positions
        positions = self._calculate_wire_positions()
        coil.coil_vertices = positions
        
        return coil
    
    def _calculate_wire_positions(self):
        """Calculate all wire positions for the C-shaped coil"""
        positions = []
        
        # Create base C-shape
        inner_radius = self.gap_width / 2
        outer_radius = inner_radius + self.width
        
        # Number of points to define the C-shape
        n_points = 100
        
        # Create points for C-shape
        theta = np.linspace(-np.pi/2, np.pi/2, n_points)
        
        # Create layers of windings
        n_layers = 2
        turns_per_layer = self.n_turns // n_layers
        
        for layer in range(n_layers):
            z_offset = layer * self.wire_diameter
            
            for turn in range(turns_per_layer):
                # Calculate radius for this turn
                r = inner_radius + (outer_radius - inner_radius) * (turn / turns_per_layer)
                
                # Create points for this turn
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = np.full_like(x, z_offset)
                
                # Add points for this turn
                turn_points = np.column_stack((x, y, z))
                positions.extend(turn_points)
                
                # Add return path
                return_x = [-x[-1], -x[0]]
                return_y = [y[-1], y[0]]
                return_z = [z_offset, z_offset]
                return_points = np.column_stack((return_x, return_y, return_z))
                positions.extend(return_points)
        
        return np.array(positions)

def verify_coil(coil):
    """Print verification information about the coil"""
    print("\nCoil Properties:")
    print(f"Position: {coil.pos}")
    print(f"Number of wire positions: {len(coil.coil_vertices)}")
    print(f"dI/dt value: {coil.didt}")
    return True

if __name__ == "__main__":
    # Create project directory if it doesn't exist
    project_dir = os.path.expanduser('~/tms_grid_project')
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    print(f"Project directory: {project_dir}")
    
    try:
        # Create coil instance with default parameters
        coil_designer = CShapedCoil()
        
        # Create coil definition
        coil = coil_designer.create_coil_definition()
        
        # Verify the coil
        verify_coil(coil)
        
        print("\nCoil design created successfully!")
        
    except Exception as e:
        print(f"\nError creating coil: {str(e)}")