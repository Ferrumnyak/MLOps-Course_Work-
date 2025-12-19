from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import sklearn


# ---------- CONFIG ----------
def load_config(config_name: str):
    config_path = Path(f"config/{config_name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл {config_path} не найден: {config_path}")
    return OmegaConf.load(config_path)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


cfg = load_config("train_rf")


# ---------- PATHS ----------
data_path = Path(cfg.data.train_lemma_csv)
model_path = Path(cfg.model.save_path)
model_path.parent.mkdir(parents=True, exist_ok=True)

# артефакт с инфой о версиях/параметрах (полезно для Docker)
meta_path = model_path.with_suffix(".meta.json")


# ---------- DATA ----------
train_df = pd.read_csv(cfg.data.train_lemma_csv)
test_df = pd.read_csv(cfg.data.test_lemma_csv)

X_train = train_df[cfg.data.text_column].astype(str)
y_train = train_df[cfg.data.target_column].astype(float)

X_test = test_df[cfg.data.text_column].astype(str)
y_test = test_df[cfg.data.target_column].astype(float)


# ---------- PIPELINE ----------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=int(cfg.tfidf.max_features),
        ngram_range=tuple(cfg.tfidf.ngram_range),
    )),
    ("rf", RandomForestRegressor(
        n_estimators=int(cfg.random_forest.n_estimators),
        random_state=int(cfg.random_forest.random_state),
        n_jobs=-1
    )),
])

# ---------- MLFLOW ----------
mlflow.set_experiment("rf_series_rating")

with mlflow.start_run():

    # params
    mlflow.log_params({
        "tfidf_max_features": int(cfg.tfidf.max_features),
        "tfidf_ngram_range": str(tuple(cfg.tfidf.ngram_range)),
        "rf_n_estimators": int(cfg.random_forest.n_estimators),
        "rf_random_state": int(cfg.random_forest.random_state),
        "split_test_size": test_size,
        "split_random_state": random_state,
        "sklearn_version": sklearn.__version__,
    })

    # train
    pipeline.fit(X_train, y_train)

    # --------- SAFETY CHECK (чтобы не было NotFittedError в сервисе) ----------
    # TfidfVectorizer после fit обязан иметь vocabulary_
    tfidf = pipeline.named_steps["tfidf"]
    check_is_fitted(tfidf, attributes=["vocabulary_"])

    # eval
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "train_rmse": rmse(y_train, y_pred_train),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "test_rmse": rmse(y_test, y_pred_test),
        "test_r2": float(r2_score(y_test, y_pred_test)),
    }

    mlflow.log_metrics(metrics)

    # save local (ЭТОТ файл потом копируешь в mlservice/models/)
    joblib.dump(pipeline, model_path)

    # meta (для воспроизводимости и чтобы знать версию sklearn при обучении)
    meta = {
        "model_path": str(model_path.as_posix()),
        "data_path": str(data_path.as_posix()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "sklearn_version": sklearn.__version__,
        "params": {
            "tfidf_max_features": int(cfg.tfidf.max_features),
            "tfidf_ngram_range": list(cfg.tfidf.ngram_range),
            "rf_n_estimators": int(cfg.random_forest.n_estimators),
            "rf_random_state": int(cfg.random_forest.random_state),
            },
            "metrics": metrics,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # log model to mlflow
    mlflow.sklearn.log_model(pipeline, artifact_path="random_forest_model")

    # also log local artifacts
    mlflow.log_artifact(str(model_path), artifact_path="model_file")
    mlflow.log_artifact(str(meta_path), artifact_path="model_file")
    mlflow.log_artifact(str(Path(f"config/train_rf.yaml")), artifact_path="config")

    print(f"✅ Model saved to {model_path}")
    print("✅ Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
