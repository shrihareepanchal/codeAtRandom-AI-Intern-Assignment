import sys
from pathlib import Path

# Allow "src" imports when run from project root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.indexing import build_index  # noqa: E402


if __name__ == "__main__":
    print("Building index over documents in ./data/docs ...")
    engine = build_index()
    print(f"Indexed {len(engine.doc_ids)} documents.")
    print("Done.")
