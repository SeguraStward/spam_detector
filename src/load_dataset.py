"""Carga y union de datasets bilingues (ingles + espanol)."""
import csv
from pathlib import Path

import pandas as pd


def load_folder_dataset(root: Path, mapping: dict[str, str]) -> pd.DataFrame:
    """Carga datasets organizados como carpetas por clase (ej. SpamAssassin/Enron).

    `mapping` mapea nombre de carpeta -> etiqueta canonica ('spam' / 'ham').
    """
    root = Path(root)
    rows = []
    for folder, label in mapping.items():
        folder_path = root / folder
        if not folder_path.exists():
            continue
        for filepath in folder_path.iterdir():
            if not filepath.is_file():
                continue
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            rows.append({"text": text, "label": label, "lang": "en"})
    return pd.DataFrame(rows)


def load_tsv_dataset(path: Path, label_col: str = "label", text_col: str = "text",
                     lang: str = "unknown") -> pd.DataFrame:
    """Carga TSV con columnas etiqueta + texto (formato SMS Spam Collection)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["text", "label", "lang"])
    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        names=[label_col, text_col],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        on_bad_lines="skip",
    )
    df = df.rename(columns={label_col: "label", text_col: "text"})
    df["label"] = df["label"].str.lower().replace({"ham": "ham", "spam": "spam"})
    df = df[df["label"].isin(["ham", "spam"])]
    df["lang"] = lang
    return df[["text", "label", "lang"]].reset_index(drop=True)


def load_all(data_dir: str | Path = "data") -> pd.DataFrame:
    """Une todas las fuentes disponibles bajo `data/`.

    Fuentes esperadas:
      - data/raw/spam, data/raw/easy_ham         (SpamAssassin estilo, EN)
      - data/raw/enron/spam, data/raw/enron/ham  (Enron-Spam, EN)
      - data/sms_spam_en.tsv                     (UCI SMS Spam, EN)
      - data/spanish_spam.tsv                    (curado, ES)
    """
    data_dir = Path(data_dir)
    frames = []

    spamassassin = load_folder_dataset(
        data_dir / "raw",
        {"spam": "spam", "easy_ham": "ham", "hard_ham": "ham"},
    )
    if not spamassassin.empty:
        frames.append(spamassassin)

    enron = load_folder_dataset(
        data_dir / "raw" / "enron",
        {"spam": "spam", "ham": "ham"},
    )
    if not enron.empty:
        frames.append(enron)

    sms_en = load_tsv_dataset(data_dir / "sms_spam_en.tsv", lang="en")
    if not sms_en.empty:
        frames.append(sms_en)

    sms_es = load_tsv_dataset(data_dir / "spanish_spam.tsv", lang="es")
    if not sms_es.empty:
        frames.append(sms_es)

    if not frames:
        raise FileNotFoundError(
            "No se encontraron datasets bajo data/. "
            "Ejecuta primero: python -m src.download_data"
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_all()
    print(f"Total: {len(df)} mensajes")
    print(df.groupby(["lang", "label"]).size().unstack(fill_value=0))
