"""Repository locations. New runs do not overwrite published reference results."""
from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = Path(os.environ.get("PHYSYM_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")).expanduser().resolve()
REFERENCE_DIR = PROJECT_ROOT / "results" / "reference"
OUTPUT_DIR = Path(os.environ.get("PHYSYM_OUTPUT_DIR", PROJECT_ROOT / "results" / "generated")).expanduser().resolve()

def ensure_output():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
