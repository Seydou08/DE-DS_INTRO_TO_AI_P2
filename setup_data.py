from pathlib import Path
import shutil
import kagglehub
from kagglehub.exceptions import KaggleApiHTTPError

DATASET_SLUG = "sobhanmoosavi/us-accidents"

project_root = Path(__file__).resolve().parent
raw_dir = project_root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

try:
    download_dir = Path(kagglehub.dataset_download(DATASET_SLUG, force_download=True))
    print(f"Downloaded to: {download_dir}")
except KaggleApiHTTPError as e:
    raise RuntimeError(
        "Kaggle access failed. Check slug, API key location, and dataset consent on Kaggle."
    ) from e

all_files = list(download_dir.rglob("*"))
print(f"Files found: {[f.name for f in all_files if f.is_file()]}")

csv_files = [f for f in all_files if f.suffix.lower() == ".csv"]
if not csv_files:
    raise FileNotFoundError(f"No CSV found in: {download_dir}")

for src in csv_files:
    dst = raw_dir / src.name
    shutil.copy2(src, dst)
    print(f"Copied: {src} -> {dst}")