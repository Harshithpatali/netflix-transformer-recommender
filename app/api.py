from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from functools import lru_cache

from src.inference import InferenceEngine

app = FastAPI()

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# REQUEST MODEL
# =====================================================

class RecommendationRequest(BaseModel):
    watched: list[str]

# =====================================================
# LOAD MODEL ONCE
# =====================================================

@lru_cache()
def get_engine():
    return InferenceEngine()

# =====================================================
# ROUTES
# =====================================================

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/recommend")
def recommend(req: RecommendationRequest):

    if len(req.watched) == 0:

        raise HTTPException(
            status_code=400,
            detail="Please select at least one movie"
        )

    engine = get_engine()

    try:

        recommendations, _ = engine.recommend(
            req.watched
        )

        return {
            "recommendations": recommendations
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )