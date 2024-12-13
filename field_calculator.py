from simnibs import sim_struct, run_simnibs, read_msh, mesh_io
import numpy as np
import os
import shutil
import datetime

def create_clean_output_dir(base_path):
    """Create a new, clean output directory for simulation results"""
    # Create timestamp-based directory name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_path, f"simulation_{timestamp}")
    
    # Create the directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    return output_dir

def get_simnibs_paths():
    """Get the paths to SimNIBS resources"""
    # Get path to SimNIBS package
    simnibs_path = os.path.dirname(os.path.dirname(os.path.dirname(sim_struct.__file__)))
    
    # Define paths
    paths = {
        'coil': os.path.join(simnibs_path, 'simnibs', 'resources', 'coil_models', 
                            'legacy_and_other', 'Magstim_70mm_Fig8.ccd'),
        'head_mesh': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 'simnibs4_examples', 'm2m_ernie', 'ernie.msh')
    }
    
    return paths

def setup_simulation():
    """Set up the simulation environment"""
    # Initialize session
    session = sim_struct.SESSION()
    
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    head_mesh = os.path.join(base_dir, 'simnibs4_examples', 'm2m_ernie', 'ernie.msh')
    simulations_dir = os.path.join(base_dir, 'simulations')
    
    # Create clean output directory
    output_dir = create_clean_output_dir(simulations_dir)
    
    # Set paths in session
    session.fnamehead = head_mesh
    session.pathfem = output_dir
    
    # Verify head mesh exists
    if not os.path.exists(head_mesh):
        raise FileNotFoundError(f"Head mesh not found at: {head_mesh}")
    
    print(f"Head mesh: {session.fnamehead}")
    print(f"Output directory: {session.pathfem}")
    
    # Get path to coil file
    coil_path = os.path.join('/Users/ajithsenthil/Applications/SimNIBS-4.1/simnibs_env/lib/python3.9/site-packages/simnibs/resources/coil_models/legacy_and_other/Magstim_70mm_Fig8.ccd')
    
    return session, {'coil': coil_path}

def create_tms_list(session, paths):
    """Create and configure TMS simulation"""
    # Initialize TMS list
    tmslist = session.add_tmslist()
    
    # Set coil file
    tmslist.fnamecoil = paths['coil']
    
    # Add coil position
    pos = tmslist.add_position()
    pos.centre = 'C3'  # Position over motor cortex
    pos.pos_ydir = 'Cz'  # Pointing towards vertex
    pos.didt = 1.0  # Relative stimulator intensity
    
    print(f"Coil file: {tmslist.fnamecoil}")
    print(f"Position: {pos.centre} pointing to {pos.pos_ydir}")
    
    return tmslist

def run_field_calculation(session):
    """Run the field calculation"""
    try:
        print("\nStarting simulation...")
        results = run_simnibs(session)
        print("Simulation completed successfully")
        return results
    
    except Exception as e:
        print(f"\nError during simulation: {str(e)}")
        print("\nSession details:")
        print(f"Head mesh exists: {os.path.exists(session.fnamehead)}")
        print(f"Output directory exists: {os.path.exists(session.pathfem)}")
        
        if hasattr(session, 'poslist'):
            print("\nTMS List settings:")
            for i, pos in enumerate(session.poslist):
                print(f"Position {i}:", pos.__dict__)
        raise

def analyze_results(session):
    """Analyze the field calculation results"""
    try:
        # Get the result mesh file path
        sim_dir = session.pathfem
        result_files = [f for f in os.listdir(sim_dir) if f.endswith('_scalar.msh')]
        
        if not result_files:
            raise FileNotFoundError("No result files found")
            
        result_file = os.path.join(sim_dir, result_files[0])
        print(f"\nAnalyzing results from: {result_file}")
        
        # Read the mesh
        mesh = mesh_io.read_msh(result_file)
        
        print("\nMesh Information:")
        print(f"Number of elements: {mesh.elm.nr}")
        print(f"Number of nodes: {mesh.nodes.nr}")
        
        # Get field data as numpy arrays
        e_mag = mesh.field['magnE'].value
        e_field = mesh.field['E'].value
        
        print("\nField Data Summary:")
        print(f"E-field shape: {e_field.shape}")
        print(f"E-magnitude shape: {e_mag.shape}")
        
        # Scale values from MV/m to V/m
        e_mag = e_mag * 1e6
        e_field = e_field * 1e6
        
        print("\nRaw Field Statistics:")
        print(f"E-field magnitude range: {np.min(e_mag):.2f} to {np.max(e_mag):.2f} V/m")
        print(f"Mean E-field magnitude: {np.mean(e_mag):.2f} V/m")
        
        # Get tissue information
        tissue_tags = mesh.elm.tag2
        unique_tags = np.unique(tissue_tags)
        print(f"\nTissue Tags Found: {unique_tags}")
        
        # Find gray matter elements (tag 2)
        gm_mask = tissue_tags == 2
        print(f"Number of Gray Matter elements: {np.sum(gm_mask)}")
        
        # Calculate field statistics for gray matter
        e_mag_gm = e_mag[gm_mask]
        max_e = np.max(e_mag_gm)
        mean_e = np.mean(e_mag_gm)
        
        # Calculate volumes
        volumes = mesh.elements_volumes_and_areas()
        gm_volumes = volumes[gm_mask]
        total_gm_volume = np.sum(gm_volumes)
        
        # Calculate stimulation volumes at different thresholds
        thresholds = [0.25, 0.5, 0.75]  # 25%, 50%, 75% of max field
        stim_volumes = {}
        for threshold in thresholds:
            thresh_value = threshold * max_e
            stim_volumes[threshold] = np.sum(gm_volumes[e_mag_gm > thresh_value])
        
        print("\nField Analysis Results:")
        print(f"Maximum E-field in GM: {max_e:.2f} V/m")
        print(f"Mean E-field in GM: {mean_e:.2f} V/m")
        print(f"\nStimulation Volumes:")
        for threshold, volume in stim_volumes.items():
            print(f"Volume above {threshold*100}% max: {volume:.2f} mm³")
        
        # Calculate percentiles
        percentiles = [99.9, 99.0, 95.0, 75.0, 50.0]
        print("\nField Percentiles:")
        for p in percentiles:
            value = np.percentile(e_mag_gm, p)
            print(f"{p}th percentile: {value:.2f} V/m")
        
        # Get spatial distribution
        centers = mesh.elements_baricenters()
        gm_centers = centers[gm_mask]
        
        spatial_spread = {
            'x': np.ptp(gm_centers[:,0]),
            'y': np.ptp(gm_centers[:,1]),
            'z': np.ptp(gm_centers[:,2])
        }
        
        print("\nSpatial Distribution:")
        print(f"Field spread X: {spatial_spread['x']:.2f} mm")
        print(f"Field spread Y: {spatial_spread['y']:.2f} mm")
        print(f"Field spread Z: {spatial_spread['z']:.2f} mm")
        
        # Save detailed results
        output_file = os.path.join(sim_dir, 'field_analysis.txt')
        with open(output_file, 'w') as f:
            f.write("TMS Field Analysis Results\n")
            f.write("=========================\n\n")
            
            f.write("Raw Field Statistics:\n")
            f.write(f"E-field range: {np.min(e_mag):.2f} to {np.max(e_mag):.2f} V/m\n")
            f.write(f"Mean field: {np.mean(e_mag):.2f} V/m\n\n")
            
            f.write("Gray Matter Analysis:\n")
            f.write(f"Volume analyzed: {total_gm_volume:.2f} mm³\n")
            f.write(f"Maximum E-field: {max_e:.2f} V/m\n")
            f.write(f"Mean E-field: {mean_e:.2f} V/m\n\n")
            
            f.write("Stimulation Volumes:\n")
            for threshold, volume in stim_volumes.items():
                f.write(f"Volume above {threshold*100}% max: {volume:.2f} mm³\n")
            
            f.write("\nField Percentiles:\n")
            for p in percentiles:
                f.write(f"{p}th percentile: {np.percentile(e_mag_gm, p):.2f} V/m\n")
            
            f.write("\nSpatial Distribution:\n")
            f.write(f"Field spread X: {spatial_spread['x']:.2f} mm\n")
            f.write(f"Field spread Y: {spatial_spread['y']:.2f} mm\n")
            f.write(f"Field spread Z: {spatial_spread['z']:.2f} mm\n")
        
        return {
            'max_e_field': max_e,
            'mean_e_field': mean_e,
            'stim_volumes': stim_volumes,
            'spatial_spread': spatial_spread,
            'e_field_data': e_mag_gm
        }
        
    except Exception as e:
        print(f"\nError in analysis: {str(e)}")
        raise



def main():
    try:
        print("Setting up simulation...")
        session, paths = setup_simulation()
        
        print("\nCreating TMS configuration...")
        tmslist = create_tms_list(session, paths)
        
        print("\nRunning field calculations...")
        run_simnibs(session)
        
        print("\nAnalyzing results...")
        analysis = analyze_results(session)
        
        print("\nWriting results to file...")
        output_file = os.path.join(session.pathfem, 'field_analysis.txt')
        with open(output_file, 'w') as f:
            f.write("Analysis complete. See field_analysis.txt for detailed results.\n")
        
        print("\nSimulation and analysis complete!")
        print(f"Results saved to: {output_file}")
        
        # Explicitly close any open resources
        import gc
        gc.collect()
        
        # Force exit
        import sys
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        print("\nSimNIBS version:", sim_struct.__version__)
        print("\nCurrent working directory:", os.getcwd())
        sys.exit(1)

if __name__ == "__main__":
    main()