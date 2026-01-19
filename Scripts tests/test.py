from collections import Counter
from pathlib import Path
import pandas as pd

Data_path = Path("../Données")
codecommune_dtypes = Counter()

for file in Data_path.rglob("*.csv"):
    if file.name.startswith("._"):
        continue
    try:
        df = pd.read_csv(file, nrows=200, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file, nrows=200, encoding="latin1")

    if "codecommune" in df.columns:
        codecommune_dtypes[str(df["codecommune"].dtype)] += 1

print(dict(codecommune_dtypes))

