import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split


# Пути к данным
DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")

# Параметры разбиения
RANDOM_STATE = 42
TEST_SIZE = 0.2


def clean_text(text: str) -> str:
    """
    Базовая нормализация текста:
    - приведение к нижнему регистру
    - удаление спецсимволов
    - удаление лишних пробелов
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_all_csvs(data_dir: Path) -> pd.DataFrame:
    """
    Загружает все CSV файлы из директории,
    добавляет признак сериала и объединяет в один DataFrame
    """
    dfs = []

    for csv_file in data_dir.glob("*.csv"):
        series_name = csv_file.stem

        df = pd.read_csv(csv_file)

        required_columns = {"title", "review", "rating"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"Файл {csv_file.name} не содержит обязательные колонки: "
                f"{required_columns}"
            )

        # Удаляем строки с пропусками
        df = df.dropna(subset=["title", "review", "rating"])

        # Формируем текст
        raw_text = df["title"] + ". " + df["review"]
        df["text"] = raw_text.apply(clean_text)

        # Приводим рейтинг к числу
        df["rating"] = df["rating"].astype(float)

        # Название сериала
        df["series"] = series_name

        dfs.append(df[["text", "rating", "series"]])

    return pd.concat(dfs, ignore_index=True)


def main():
    # Загружаем и объединяем данные
    df = load_all_csvs(DATA_RAW_DIR)

    # Train / Test split со стратификацией по сериалам
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["series"]
    )

    # Сохраняем
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / "test.csv", index=False)

    # Логи
    print("Data preparation completed.")
    print(f"Total samples: {len(df)}")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print("\nSeries distribution:")
    print(df["series"].value_counts())


if __name__ == "__main__":
    main()
