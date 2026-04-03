import pandas as pd
from pathlib import Path


def load_dataset(data_path )-> pd.DataFrame: 
    data_path = Path(data_path)
    rows = []

    for label, folder in [("spam", "spam"), ("ham", "easy_ham")]:
        folder_path = data_path / folder
        for filepath in folder_path.iterdir():
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            rows.append({"label": label, "text": text})

     
    return pd.DataFrame(rows)

