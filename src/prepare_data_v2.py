import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- spaCy (лемматизация) ---
import spacy

# Пути к данным
DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")  # отдельная версия датасета

# Параметры разбиения
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Загружаем модель spaCy (англ.)
# Установить: python -m spacy download en_core_web_sm
NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def basic_clean(text: str) -> str:
    """
    Базовая нормализация текста:
    - нижний регистр
    - удаление спецсимволов
    - удаление лишних пробелов
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_text(text: str) -> str:
    """
    Лемматизация (англ.) через spaCy:
    - выкидываем стоп-слова и пунктуацию
    - берём lemma_
    """
    doc = NLP(text)
    lemmas = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        # убираем совсем короткие токены типа "s"
        lemma = token.lemma_.strip()
        if len(lemma) < 2:
            continue
        lemmas.append(lemma)
    return " ".join(lemmas)


def clean_and_lemmatize(text: str) -> str:
    text = basic_clean(text)
    return lemmatize_text(text)


def load_all_csvs(data_dir: Path) -> pd.DataFrame:
    """
    Загружает все CSV из data/raw, формирует text, чистит + лемматизирует,
    добавляет серию (series) и объединяет всё в один DataFrame.
    """
    dfs = []

    for csv_file in data_dir.glob("*.csv"):
        series_name = csv_file.stem
        df = pd.read_csv(csv_file)

        required_columns = {"title", "review", "rating"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"Файл {csv_file.name} не содержит обязательные колонки: {required_columns}"
            )

        df = df.dropna(subset=["title", "review", "rating"])

        raw_text = df["title"].astype(str) + ". " + df["review"].astype(str)
        df["text"] = raw_text.apply(clean_and_lemmatize)

        df["rating"] = df["rating"].astype(float)
        df["series"] = series_name

        dfs.append(df[["text", "rating", "series"]])

    return pd.concat(dfs, ignore_index=True)


def main():
    df = load_all_csvs(DATA_RAW_DIR)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["series"]
    )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_PROCESSED_DIR / "train_lemma.csv", index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / "test_lemma.csv", index=False)

    print("Data preparation (with lemmatization) completed.")
    print(f"Total samples: {len(df)}")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print("\nSeries distribution:")
    print(df["series"].value_counts())


if __name__ == "__main__":
    main()
