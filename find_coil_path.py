"""Locate TMS coil model files (.tcd / .ccd) in standard locations.

Provides ``get_coil_path()`` which returns the path to the best available
coil file — preferring the custom Omnidream miniature coil, then falling
back to SimNIBS built-in models.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_simnibs_root() -> str:
    """Find SimNIBS installation directory from the Python executable."""
    python_path = sys.executable
    simnibs_root = os.path.dirname(os.path.dirname(python_path))
    return simnibs_root


def find_coil_files(start_path: str, extensions: tuple[str, ...] = (".tcd", ".ccd")) -> list[str]:
    """Search for coil files recursively under *start_path*."""
    coil_files: list[str] = []
    if not os.path.isdir(start_path):
        return coil_files
    for root, _dirs, files in os.walk(start_path):
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                coil_files.append(os.path.join(root, f))
    return coil_files


def get_coil_path(preferred_name: str = "c_shaped_miniature_v1.tcd") -> str:
    """Return the path to the best available coil model file.

    Search order:
      1. ``Omnidream/coil_models/<preferred_name>`` (custom miniature coil)
      2. Any ``.tcd`` or ``.ccd`` in ``Omnidream/coil_models/``
      3. SimNIBS built-in coil directories

    Raises
    ------
    FileNotFoundError
        If no coil file can be found anywhere.
    """
    here = Path(__file__).resolve().parent

    # 1. Preferred custom coil
    custom_path = here / "coil_models" / preferred_name
    if custom_path.exists():
        return str(custom_path)

    # 2. Any coil in local coil_models/
    local_dir = here / "coil_models"
    if local_dir.is_dir():
        local_files = find_coil_files(str(local_dir))
        if local_files:
            return local_files[0]

    # 3. SimNIBS installation directories
    possible_roots = [
        os.path.expanduser("~/Applications/SimNIBS-4.5"),
        os.path.expanduser("~/Applications/SimNIBS-4.1"),
        os.path.expanduser("~/SimNIBS"),
        os.path.expanduser("~/AppData/Local/SimNIBS"),
    ]
    try:
        possible_roots.append(find_simnibs_root())
    except Exception:
        pass

    for root in possible_roots:
        if os.path.isdir(root):
            found = find_coil_files(root)
            if found:
                return found[0]

    raise FileNotFoundError(
        "No coil model file (.tcd/.ccd) found. "
        "Place a coil file in Omnidream/coil_models/ or install SimNIBS."
    )


def main() -> None:
    """CLI entry point: print all discoverable coil files."""
    try:
        import simnibs
        print("SimNIBS version:", simnibs.__version__)
        print("SimNIBS package location:", os.path.dirname(simnibs.__file__))
    except ImportError:
        print("SimNIBS not importable in this Python environment.")

    print(f"\nDefault coil path: {get_coil_path()}")

    here = Path(__file__).resolve().parent
    search_roots = [
        str(here / "coil_models"),
        os.path.expanduser("~/Applications/SimNIBS-4.5"),
        os.path.expanduser("~/Applications/SimNIBS-4.1"),
        os.path.expanduser("~/SimNIBS"),
    ]

    all_files: list[str] = []
    for root in search_roots:
        all_files.extend(find_coil_files(root))

    if all_files:
        print(f"\nFound {len(all_files)} coil file(s):")
        for i, f in enumerate(all_files, 1):
            print(f"  {i}. {f}")
    else:
        print("\nNo coil files found in standard locations.")


if __name__ == "__main__":
    main()