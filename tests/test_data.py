import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
KEYS = ["source_file", "window_id"]


def test_processed_tables_are_aligned():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    expected = None
    for condition in manifest["conditions"]:
        path = DATA / f"{condition}.csv.gz"
        assert path.exists(), path
        frame = pd.read_csv(path)
        assert frame.shape == (4752, 102)
        assert frame[KEYS].duplicated().sum() == 0
        keys = frame[KEYS].sort_values(KEYS).reset_index(drop=True)
        if expected is None:
            expected = keys
        else:
            assert expected.equals(keys), condition
