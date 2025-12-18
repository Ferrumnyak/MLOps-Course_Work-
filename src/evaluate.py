import pandas as pd
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Пути
TEST_PATH = "data/processed/test.csv"
RF_MODEL_PATH = "models/random_forest.pkl"
BERT_MODEL_DIR = Path("models/bert_model")
OUTPUT_PATH = Path("inference/predictions.csv")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

# Загружаем тест
df_test = pd.read_csv(TEST_PATH)
X_test_texts = df_test["text"]
y_true = df_test["rating"]

# --- RandomForest ---
rf_model = joblib.load(RF_MODEL_PATH)
rf_preds = rf_model.predict(X_test_texts)

# --- BERT ---
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
bert_model.eval()

# Подготовка токенов
encodings = tokenizer(list(X_test_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')
with torch.no_grad():
    outputs = bert_model(**encodings)
    bert_preds = outputs.logits.squeeze().numpy()  # регрессия, logits = предсказанный рейтинг

# Метрики
def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} evaluation:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

print_metrics("RandomForest", y_true, rf_preds)
print_metrics("BERT", y_true, bert_preds)

# Сохраняем предсказания
df_test["rf_pred"] = rf_preds
df_test["bert_pred"] = bert_preds
df_test.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")
