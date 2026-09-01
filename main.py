import os
import pickle
from typing import Literal

import pandas as pd
from google import genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="SugarTalk Diabetes Screening API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model artifacts ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_screening_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "screening_threshold.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(THRESHOLD_PATH, "rb") as f:
    flag_threshold = pickle.load(f)

# --- Gemini Client ---
gemini_client = genai.Client()
LLM_MODEL = "gemini-2.5-flash"


# --- Schemas ---
class ScreeningInput(BaseModel):
    gender: Literal["Female", "Male", "Other"]
    age: float = Field(..., ge=0, le=120)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    smoking_history: Literal["never", "No Info", "current", "former", "ever", "not current"]
    bmi: float = Field(..., ge=10, le=80)


class ScreeningResult(BaseModel):
    probability: float
    risk_tier: str
    flagged: bool
    recommendations: str


# --- Helpers ---
def risk_tier_from_probability(probability: float, threshold: float) -> str:
    if probability < 0.15:
        return "Low"
    elif probability < threshold:
        return "Moderate"
    else:
        return "High"


def build_recommendation_prompt(data: ScreeningInput, probability: float, tier: str) -> str:
    return f"""A user completed a diabetes risk screening (this is NOT a diagnosis) with these self-reported details:
- Gender: {data.gender}
- Age: {data.age}
- Hypertension: {"Yes" if data.hypertension else "No"}
- Heart disease: {"Yes" if data.heart_disease else "No"}
- Smoking history: {data.smoking_history}
- BMI: {data.bmi}

Model output: estimated risk probability {probability:.0%}, risk tier: {tier}.

Write a short, warm, non-alarming message for the user with:
1. One plain-language sentence on what this risk tier means.
2. 3-4 practical lifestyle tips relevant to their specific inputs (diet, activity, smoking, weight -- only mention what's relevant to them).
3. A clear line recommending a confirmatory HbA1c or fasting glucose test at a clinic -- make this firmer if the tier is Moderate or High.

Keep it under 180 words, speak directly to the user ("you"), do not diagnose, and do not suggest any medication or dosages."""


# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ScreeningResult)
async def predict(data: ScreeningInput):
    try:
        row = pd.DataFrame([{
            "gender": data.gender,
            "age": data.age,
            "hypertension": data.hypertension,
            "heart_disease": data.heart_disease,
            "smoking_history": data.smoking_history,
            "bmi": data.bmi,
        }])

        probability = float(model.predict_proba(row)[:, 1][0])
        tier = risk_tier_from_probability(probability, flag_threshold)
        flagged = bool(probability >= flag_threshold)

        recommendations = "Consult a licensed medical provider for confirmatory laboratory testing."
        try:
            prompt = build_recommendation_prompt(data, probability, tier)
            response = gemini_client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
            )
            if response.text:
                recommendations = response.text
        except Exception as llm_err:
            print(f"Gemini generation error: {llm_err}")

        return ScreeningResult(
            probability=round(probability, 4),
            risk_tier=tier,
            flagged=flagged,
            recommendations=recommendations,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))