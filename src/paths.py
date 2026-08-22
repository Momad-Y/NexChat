from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "imgs"


def asset_path(filename: str) -> Path:
    """Resolve a filename inside imgs/, independent of process CWD."""
    return ASSETS_DIR / filename
