from pathlib import Path

# IR/Project/ root (3 levels up from code/utils/paths.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SCRAPED_DATA_DIR = PROJECT_ROOT / "scraped_data"
CHALLENGES_DIR = SCRAPED_DATA_DIR / "challenges"
CONTESTS_DIR = SCRAPED_DATA_DIR / "contests"
ANNOTATIONS_DIR = PROJECT_ROOT / "openai_annotations"
PRECOMPUTED_DIR = PROJECT_ROOT / "precomputed"
RESULTS_DIR = PROJECT_ROOT / "results"
