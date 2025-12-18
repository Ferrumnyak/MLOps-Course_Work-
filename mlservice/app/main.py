from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
from typing import Optional
import joblib


# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlservice")


# ---------- API ----------
app = FastAPI(
    title="Series Rating Prediction API",
    description="API для предсказания рейтинга сериала по тексту отзыва (RandomForest + TF-IDF)",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    rating: float
    model_used: str


class ModelManager:
    def __init__(self):
        self.model = None
        self.model_loaded: bool = False
        self.model_path: Optional[str] = None

    def load_model(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))  # /app/app
            # app/main.py -> base_dir = .../mlservice/app
            # model лежит в .../mlservice/models/random_forest.pkl
            model_path = os.path.abspath(os.path.join(base_dir, "..", "models", "random_forest.pkl"))  # /app/models/random_forest.pkl
            self.model_path = model_path

            logger.info("🤖 Loading RandomForest model...")
            logger.info(f"📁 Model path: {model_path}")

            if not os.path.exists(model_path):
                models_dir = os.path.abspath(os.path.join(base_dir, "..", "models"))
                existing = os.listdir(models_dir) if os.path.exists(models_dir) else []
                raise FileNotFoundError(
                    f"Model file not found: {model_path}. "
                    f"models dir: {models_dir}, contents: {existing}"
                )

            self.model = joblib.load(model_path)

            # Быстрая sanity-проверка: у sklearn Pipeline есть predict
            if not hasattr(self.model, "predict"):
                raise TypeError("Loaded object has no predict() method. Is it a sklearn model/pipeline?")

            self.model_loaded = True
            logger.info("✅ Model loaded successfully")

        except Exception as e:
            logger.exception(f"❌ Failed to load model: {e}")
            self.model_loaded = False
            self.model = None

    def predict(self, text: str) -> float:
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model is not loaded")

        # sklearn Pipeline ожидает список строк
        pred = self.model.predict([text])

        # pred обычно np.array([value])
        rating = float(pred[0])
        return rating


model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    model_manager.load_model()


@app.get("/")
async def root():
    return {
        "message": "Series Rating Prediction API",
        "status": "running",
        "model_loaded": model_manager.model_loaded,
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_manager.model_loaded else "loading",
        "model_loaded": model_manager.model_loaded,
        "model_path": model_manager.model_path,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if not model_manager.model_loaded:
            raise HTTPException(status_code=503, detail="Model is not loaded yet")

        text = (request.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Field 'text' must be non-empty")

        rating = model_manager.predict(text)

        return PredictionResponse(
            text=request.text,
            rating=rating,
            model_used="random_forest"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # для локального запуска без докера
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
