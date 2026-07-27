import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed" / "splits.csv"

df = pd.read_csv(DATA_DIR)
pivot = df.groupby(['qari_id', 'split']).size().unstack(fill_value=0)
print(pivot)
print()
print('Qaris with 0 in any split:')
print(pivot[(pivot == 0).any(axis=1)])