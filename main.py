"""
SugarTalk Backend
==================
Loads the trained diabetes screening model + threshold, exposes a /predict
endpoint that:
    1. Runs the sklearn model to get a risk probability
    2. Buckets that into a risk tier (Low / Moderate / High)
    3. Calls the Gemini API to turn that into a short, plain-language
       set of tips + a recommendation to get confirmatory lab testing

Environment variables expected (set these in Render, not in code):
    GEMINI_API_KEY -> your Gemini API key from aistudio.google.com/apikey
"""

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
# Allows Lovable and local development clients to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model artifacts once at startup ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_screening_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "screening_threshold.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(THRESHOLD_PATH, "rb") as f:
    flag_threshold = pickle.load(f)

# --- Gemini Client (Safe Startup) ---
# Reads GEMINI_API_KEY or GOOGLE_API_KEY from environment variables
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=api_key) if api_key else None
LLM_MODEL = "gemini-2.5-flash"


# --- Request / response schemas ---
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
    """
    Three-tier bucket for display purposes. The binary `flagged` field
    uses the model's own recall-tuned threshold.
    """
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
@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SugarTalk API",
        "gemini_connected": gemini_client is not None
    }


@app.post("/predict", response_model=ScreeningResult)
async def predict(data: ScreeningInput):
    try:
        # 1. Prepare tabular feature row
        row = pd.DataFrame([{
            "gender": data.gender,
            "age": data.age,
            "hypertension": data.hypertension,
            "heart_disease": data.heart_disease,
            "smoking_history": data.smoking_history,
            "bmi": data.bmi,
        }])

        # 2. Run Random Forest Inference
        probability = float(model.predict_proba(row)[:, 1][0])
        tier = risk_tier_from_probability(probability, flag_threshold)
        flagged = bool(probability >= flag_threshold)

        # 3. LLM Recommendation Synthesis
        recommendations = (
            "We recommend consulting a healthcare professional for standard confirmatory "
            "laboratory testing (e.g., Fasting Blood Glucose or HbA1c)."
        )

        if gemini_client:
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
