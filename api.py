from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import time
from typing import List, Literal

MODEL_PATH = os.path.join(os.path.dirname(__file__), "mental_health_pipeline.joblib")
pipeline = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

app = FastAPI(
    title="Student Mental Health Prediction Microservice",
    description="REST API for real-time mental health score inference from student lifestyle habits.",
    version="1.0.0"
)

class StudentProfile(BaseModel):
    Age: int = Field(..., ge=15, le=35, description="Student age in years", json_schema_extra={"example": 21})
    Gender: Literal["Male", "Female", "Other"] = Field(..., json_schema_extra={"example": "Female"})
    Country: str = Field(..., min_length=2, json_schema_extra={"example": "India"})
    Academic_Level: Literal["High School", "Undergraduate", "Graduate"] = Field(..., json_schema_extra={"example": "Undergraduate"})
    Most_Used_Platform: str = Field(..., json_schema_extra={"example": "LinkedIn"})
    Purpose_Of_Use: Literal["Entertainment", "Education", "Networking", "News"] = Field(..., json_schema_extra={"example": "Education"})
    Avg_Daily_Usage_Hours: float = Field(..., ge=0.0, le=24.0, json_schema_extra={"example": 2.5})
    Daily_Unlocks: int = Field(..., ge=0, le=500, json_schema_extra={"example": 90})
    Study_Hours: float = Field(..., ge=0.0, le=24.0, json_schema_extra={"example": 5.5})
    Physical_Activity_Hours: float = Field(..., ge=0.0, le=24.0, json_schema_extra={"example": 2.0})
    Sleep_Hours_Per_Night: float = Field(..., ge=0.0, le=24.0, json_schema_extra={"example": 8.0})
    Stress_Level: Literal["Low", "Medium", "High", "Very High"] = Field(..., json_schema_extra={"example": "Low"})

class BatchStudentProfiles(BaseModel):
    profiles: List[StudentProfile]

class PredictionResponse(BaseModel):
    predicted_score: float
    wellbeing_category: str
    latency_ms: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    batch_latency_ms: float

def categorize_score(score: float) -> str:
    if score >= 7.5:
        return "Healthy Wellbeing"
    elif score >= 5.5:
        return "Moderate Wellbeing"
    return "Elevated Stress / At Risk"

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "service": "mental-health-inference-api"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_single(profile: StudentProfile):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not initialized")
    
    start_time = time.perf_counter()
    df_input = pd.DataFrame([profile.model_dump()])
    
    raw_pred = float(pipeline.predict(df_input)[0])
    bounded_score = round(max(1.0, min(10.0, raw_pred)), 2)
    latency = round((time.perf_counter() - start_time) * 1000, 2)
    
    return PredictionResponse(
        predicted_score=bounded_score,
        wellbeing_category=categorize_score(bounded_score),
        latency_ms=latency
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchStudentProfiles):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not initialized")
    
    start_time = time.perf_counter()
    records = [p.model_dump() for p in batch.profiles]
    df_input = pd.DataFrame(records)
    
    raw_preds = pipeline.predict(df_input)
    predictions = []
    
    for p in raw_preds:
        score = round(max(1.0, min(10.0, float(p))), 2)
        predictions.append(PredictionResponse(
            predicted_score=score,
            wellbeing_category=categorize_score(score),
            latency_ms=0.0
        ))
    
    total_latency = round((time.perf_counter() - start_time) * 1000, 2)
    return BatchPredictionResponse(
        predictions=predictions,
        total_count=len(predictions),
        batch_latency_ms=total_latency
    )
