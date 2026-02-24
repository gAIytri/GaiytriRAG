import os
from pathlib import Path

# Resolve paths relative to the project root (GaiytriRAG/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DB_PATH = str(PROJECT_ROOT / "db")
DATA_PATH = str(PROJECT_ROOT / "data")
ENV_PATH = str(PROJECT_ROOT / ".env")

# Validate paths exist
def validate_paths():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data directory not found: {DATA_PATH}")
    return True
