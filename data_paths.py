from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

def npy_path(name):
    """Return the absolute path for a cached NumPy file."""
    return DATA_DIR / name
