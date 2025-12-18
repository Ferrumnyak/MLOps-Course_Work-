from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.transformers

from datasets import Dataset
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)


# ---------------- CONFIG ----------------
def load_config(config_name: str):
    config_path = Path(f"config/{config_name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл {config_path} не найден")
    return OmegaConf.load(config_path)


cfg = load_config("train_bert")

# ---------------- PATHS ----------------
data_path = Path(cfg.data.train_csv)
model_dir = Path(cfg.model.save_dir)
log_dir = Path(cfg.model.log_dir)

model_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

# ---------------- MLFLOW ----------------
mlflow.set_experiment("bert_series_rating")

with mlflow.start_run():

    # params (из конфига)
    mlflow.log_params({
        "pretrained_model": str(cfg.tokenizer.pretrained_model),
        "max_length": int(cfg.tokenizer.max_length),
        "epochs": int(cfg.training.num_epochs),
        "batch_size": int(cfg.training.batch_size),
        "learning_rate": float(cfg.training.learning_rate),
        "train_csv": str(cfg.data.train_csv),
    })

    # ---------------- DATA ----------------
    # Можно убрать head(), когда будешь обучать на всём train.csv
    df = pd.read_csv(data_path).head(1000)
    dataset = Dataset.from_pandas(df)

    # ---------------- TOKENIZER ----------------
    tokenizer = BertTokenizer.from_pretrained(cfg.tokenizer.pretrained_model)

    def tokenize(batch):
        return tokenizer(
            batch[cfg.data.text_column],
            padding=True,
            truncation=True,
            max_length=cfg.tokenizer.max_length
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column(cfg.data.target_column, "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # ---------------- MODEL ----------------
    model = BertForSequenceClassification.from_pretrained(
        cfg.tokenizer.pretrained_model,
        num_labels=1,
        problem_type="regression"
    )

    # ---------------- TRAINING ARGS ----------------
    training_args = TrainingArguments(
        output_dir=str(model_dir.resolve()),
        logging_dir=str(log_dir.resolve()),
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,  # например "no"
        report_to=cfg.training.report_to,          # например "none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # ---------------- TRAIN ----------------
    train_output = trainer.train()
    trainer.save_model(model_dir)

    # ---------------- METRICS (после обучения) ----------------
    mlflow.log_metric("train_loss", float(train_output.training_loss))

    pred = trainer.predict(dataset)
    y_true = pred.label_ids.astype(float).reshape(-1)
    y_pred = pred.predictions.reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    mlflow.log_metric("train_mae", float(mae))
    mlflow.log_metric("train_rmse", float(rmse))
    mlflow.log_metric("train_r2", float(r2))

    # ---------------- LOG MODEL (Transformers flavor) ----------------
    # Сохраняем модель + токенизатор в MLflow как Transformers модель
    mlflow.transformers.log_model(
        transformers_model={
            "model": trainer.model,
            "tokenizer": tokenizer,
        },
        artifact_path="bert_model",
    )

    print(f"✅ BERT saved locally to: {model_dir}")
    print(f"✅ MLflow metrics -> loss: {train_output.training_loss:.4f}, "
          f"mae: {mae:.4f}, rmse: {rmse:.4f}, r2: {r2:.4f}")

print("Training finished, experiment logged in MLflow.")
