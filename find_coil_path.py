import simnibs
import os
import sys

def find_simnibs_root():
    """Find SimNIBS installation directory"""
    # Get Python executable path
    python_path = sys.executable
    simnibs_root = os.path.dirname(os.path.dirname(python_path))
    print(f"Python executable: {python_path}")
    print(f"Potential SimNIBS root: {simnibs_root}")
    return simnibs_root

def find_coil_files(start_path):
    """Search for .ccd files recursively"""
    print(f"\nSearching for coil files in: {start_path}")
    coil_files = []
    
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file.endswith('.ccd'):
                full_path = os.path.join(root, file)
                coil_files.append(full_path)
                print(f"Found coil file: {full_path}")
    
    return coil_files

def main():
    print("SimNIBS version:", simnibs.__version__)
    print("SimNIBS package location:", os.path.dirname(simnibs.__file__))
    
    # Look in common installation locations
    possible_paths = [
        os.path.expanduser('~/Applications/SimNIBS-4.1'),  # Mac
        os.path.expanduser('~/SimNIBS'),                   # Linux
        os.path.expanduser('~/AppData/Local/SimNIBS'),     # Windows
        find_simnibs_root()
    ]
    
    coil_files = []
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\nChecking directory: {path}")
            coil_files.extend(find_coil_files(path))
    
    if coil_files:
        print("\nFound coil files:")
        for i, file in enumerate(coil_files, 1):
            print(f"{i}. {file}")
    else:
        print("\nNo coil files found in standard locations.")
        
if __name__ == "__main__":
    main()