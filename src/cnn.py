from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


CONFIG_PATH = Path("config/cnn.yaml")


def load_cfg(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_metric_safe(key: str, value):
    """Log metric only if value is finite (not NaN/Inf)."""
    try:
        v = float(value)
    except Exception:
        return
    if np.isfinite(v):
        mlflow.log_metric(key, v)


def main():
    cfg = load_cfg(CONFIG_PATH)

    # --------- OPTIONAL: MLFLOW TRACKING URI ----------
    # ВАЖНО: tracking_uri нужен, только если ты реально запустил "mlflow server".
    # Если ты запускаешь "mlflow ui", лучше НЕ задавать tracking_uri (логирование будет в ./mlruns).
    # tracking_uri = cfg.get("mlflow", {}).get("tracking_uri")
    # if tracking_uri:
        # ОСТОРОЖНО: http://127.0.0.1:5000 обычно это "mlflow ui" (viewer),
        # а не "mlflow server" (tracking API). Если будут проблемы — закомментируй set_tracking_uri.
        # mlflow.set_tracking_uri(tracking_uri)

    exp_name = cfg.get("experiment", {}).get("name", "CNN_Regression")
    run_name = cfg.get("mlflow", {}).get("run_name", "cnn_regression_run")

    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=run_name):
        # --------- TAGS ----------
        tags = cfg.get("mlflow", {}).get("tags", {})
        if isinstance(tags, dict):
            mlflow.set_tags(tags)

        # --------- PARAMS ----------
        mlflow.log_params({
            "vocab_size": int(cfg["preprocessing"]["vocab_size"]),
            "max_length": int(cfg["preprocessing"]["max_length"]),
            "padding": str(cfg["preprocessing"].get("padding", "post")),
            "embedding_dim": int(cfg["model"]["embedding_dim"]),
            "conv_filters": int(cfg["model"]["conv_filters"]),
            "conv_kernel_size": int(cfg["model"]["conv_kernel_size"]),
            "dense_units": int(cfg["model"]["dense_units"]),
            "dropout_rate": float(cfg["model"]["dropout_rate"]),
            "optimizer": str(cfg["model"].get("optimizer", "adam")),
            "loss": str(cfg["model"].get("loss", "mse")),
            "epochs": int(cfg["training"]["epochs"]),
            "batch_size": int(cfg["training"]["batch_size"]),
            "test_size": float(cfg["training"]["test_size"]),
            "validation_split": float(cfg["training"].get("validation_split", 0.1)),
            "data_path": str(cfg["data"]["artifact_path"]),
            "text_column": str(cfg["data"]["text_column"]),
            "label_column": str(cfg["data"]["label_column"]),
        })

        # --------- DATA ----------
        csv_path = Path(cfg["data"]["artifact_path"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        text_col = cfg["data"]["text_column"]
        label_col = cfg["data"]["label_column"]

        df = pd.read_csv(csv_path)

        if cfg["preprocessing"].get("clean_nan", True):
            df = df.dropna(subset=[text_col, label_col])

        texts = df[text_col].astype(str).values
        labels = df[label_col].astype(float).values  # REGRESSION

        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=float(cfg["training"]["test_size"]),
            random_state=int(cfg["training"]["random_state"]),
        )

        # --------- TOKENIZE ----------
        vocab_size = int(cfg["preprocessing"]["vocab_size"])
        max_len = int(cfg["preprocessing"]["max_length"])
        padding = cfg["preprocessing"].get("padding", "post")

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)

        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(
            X_train_seq, maxlen=max_len, padding=padding, truncating=padding
        )
        X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(
            X_test_seq, maxlen=max_len, padding=padding, truncating=padding
        )

        # --------- MODEL (REGRESSION) ----------
        emb_dim = int(cfg["model"]["embedding_dim"])
        conv_filters = int(cfg["model"]["conv_filters"])
        kernel_size = int(cfg["model"]["conv_kernel_size"])
        dense_units = int(cfg["model"]["dense_units"])
        dropout = float(cfg["model"]["dropout_rate"])

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_len,)),
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim),
            tf.keras.layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(dense_units, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, name="rating")  # REGRESSION OUTPUT
        ])

        optimizer = cfg["model"].get("optimizer", "adam")
        loss = cfg["model"].get("loss", "mse")

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse")
            ],
        )

        # --------- CALLBACKS ----------
        callbacks = []
        es_cfg = cfg["training"].get("early_stopping")
        if es_cfg:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                patience=int(es_cfg.get("patience", 2)),
                restore_best_weights=bool(es_cfg.get("restore_best_weights", True)),
            ))

        # --------- TRAIN ----------
        history = model.fit(
            X_train_pad, y_train.astype("float32"),
            validation_split=float(cfg["training"].get("validation_split", 0.1)),
            epochs=int(cfg["training"]["epochs"]),
            batch_size=int(cfg["training"]["batch_size"]),
            callbacks=callbacks,
            verbose=1
        )

        # Логируем последние значения истории (с уникальными ключами!)
        if "loss" in history.history:
            log_metric_safe("cnn_train_loss", history.history["loss"][-1])
        if "mae" in history.history:
            log_metric_safe("cnn_train_mae", history.history["mae"][-1])
        if "rmse" in history.history:
            log_metric_safe("cnn_train_rmse", history.history["rmse"][-1])

        if "val_loss" in history.history:
            log_metric_safe("cnn_val_loss", history.history["val_loss"][-1])
        if "val_mae" in history.history:
            log_metric_safe("cnn_val_mae", history.history["val_mae"][-1])
        if "val_rmse" in history.history:
            log_metric_safe("cnn_val_rmse", history.history["val_rmse"][-1])

        # --------- EVAL (TEST) ----------
        test_vals = model.evaluate(X_test_pad, y_test.astype("float32"), verbose=0)
        test_dict = dict(zip(model.metrics_names, [float(v) for v in test_vals]))

        log_metric_safe("cnn_test_loss", test_dict.get("loss", np.nan))
        log_metric_safe("cnn_test_mae", test_dict.get("mae", np.nan))
        log_metric_safe("cnn_test_rmse", test_dict.get("rmse", np.nan))

        # R2 отдельно
        y_pred = model.predict(X_test_pad, verbose=0).reshape(-1)
        r2 = r2_score(y_test, y_pred)
        log_metric_safe("cnn_test_r2", r2)

        print("\nTest metrics:")
        for k, v in test_dict.items():
            print(f"  {k}: {v:.4f}")
        print(f"  r2: {r2:.4f}")

        # --------- SAVE LOCAL (.keras) ----------
        out_dir = Path("models")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cnn_regression.keras"
        model.save(out_path)
        print(f"\n✓ Model saved to: {out_path}")

        # --------- LOG MODEL & ARTIFACTS ----------
        # 1) Логируем модель как MLflow TensorFlow Model (может логировать свои метрики внутри)
        # Метрики у нас с префиксом cnn_..., поэтому конфликтов не будет.
        mlflow.tensorflow.log_model(model, artifact_path="tf_cnn_model")

        # 2) Логируем файл .keras как артефакт
        mlflow.log_artifact(str(out_path), artifact_path="models")

        # 3) Логируем конфиг для воспроизводимости
        mlflow.log_artifact(str(CONFIG_PATH), artifact_path="config")

        print("\n✓ Logged to MLflow. Artifacts URI:", mlflow.get_artifact_uri())


if __name__ == "__main__":
    main()
