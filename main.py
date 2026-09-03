"""
SugarTalk Backend
==================
Exposes:
  1. POST /predict -> ML screening probability + risk tier + Gemini recommendations
  2. POST /chat    -> Interactive diabetes Q&A assistant with high-demand retry handling
  3. GET  /health  -> Health and service readiness check
"""

import os
import pickle
import time
import traceback
from typing import Any, List, Literal, Optional

import pandas as pd
from google import genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="SugarTalk Diabetes Screening & Q&A API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model Artifacts ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_screening_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "screening_threshold.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(THRESHOLD_PATH, "rb") as f:
    flag_threshold = pickle.load(f)

# --- Gemini Client Configuration ---
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=api_key) if api_key else None

# Diverse fallback pool to bypass localized cluster 503 capacity limits
CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-pro",
    "gemini-3.6-flash",
    "gemini-3.6-flash-lite",
]


# --- Schemas: Screening ---
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


# --- Schemas: Chat ---
class ChatRequest(BaseModel):
    message: Optional[str] = None
    text: Optional[str] = None
    risk_tier: Optional[str] = "Unknown"
    history: Optional[List[Any]] = []


class ChatResponse(BaseModel):
    reply: str


# --- Helper Functions ---
def risk_tier_from_probability(probability: float, threshold: float) -> str:
    if probability < 0.15:
        return "Low"
    elif probability < threshold:
        return "Moderate"
    else:
        return "High"


def generate_with_resilience(prompt: str) -> str:
    """Tries generation across multiple model families with retries on 503/429 spikes."""
    if not gemini_client:
        return ""

    for target_model in CANDIDATE_MODELS:
        for attempt in range(2):  # Try twice per model
            try:
                resp = gemini_client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                )
                if resp and resp.text:
                    return resp.text
            except Exception as err:
                err_str = str(err)
                print(f"[WARN] Model {target_model} attempt {attempt+1} failed: {err_str}")
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    time.sleep(1.2)  # Brief delay to allow transient spike to clear
                else:
                    break  # Not a transient capacity issue, skip to next candidate model
    return ""


def build_recommendation_prompt(data: ScreeningInput, probability: float, tier: str) -> str:
    return f"""A user completed a diabetes risk screening (NOT a diagnosis) with these inputs:
- Gender: {data.gender}
- Age: {data.age}
- Hypertension: {"Yes" if data.hypertension else "No"}
- Heart disease: {"Yes" if data.heart_disease else "No"}
- Smoking history: {data.smoking_history}
- BMI: {data.bmi}

Model output: estimated risk probability {probability:.0%}, risk tier: {tier}.

Write a short, warm, supportive message for the user with:
1. One plain-language sentence explaining what this risk tier indicates.
2. 3-4 practical lifestyle tips relevant to their inputs (nutrition, activity, smoking cessation, weight management).
3. A clear recommendation for a confirmatory fasting glucose or HbA1c lab test at a clinic.

Keep it under 180 words, address the user directly ("you"), do not diagnose, and do not suggest any medication or dosages."""


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

        default_recs = (
            "We recommend consulting a healthcare professional for standard confirmatory "
            "laboratory testing (such as a Fasting Blood Glucose or HbA1c test)."
        )

        prompt = build_recommendation_prompt(data, probability, tier)
        llm_output = generate_with_resilience(prompt)
        recommendations = llm_output if llm_output else default_recs

        return ScreeningResult(
            probability=round(probability, 4),
            risk_tier=tier,
            flagged=flagged,
            recommendations=recommendations,
        )

    except Exception as e:
        print("[ERROR] Exception in /predict:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(req: ChatRequest):
    user_query = req.message or req.text
    if not user_query:
        raise HTTPException(status_code=400, detail="Missing message or text in request payload.")

    try:
        history_lines = []
        if req.history:
            for item in req.history[-6:]:
                if isinstance(item, dict):
                    role = item.get("role", "User")
                    content = item.get("content") or item.get("text") or ""
                    history_lines.append(f"{str(role).capitalize()}: {content}")
                elif hasattr(item, "role") and hasattr(item, "content"):
                    history_lines.append(f"{str(item.role).capitalize()}: {item.content}")

        formatted_history = "\n".join(history_lines)

        system_instruction = f"""
You are SugarTalk Assistant, a supportive, certified diabetes educator and wellness guide.

Patient Context:
- User's assessed screening risk level: {req.risk_tier or "Not screened yet"}

Guidelines:
1. Answer questions clearly in accessible, encouraging language.
2. Focus on nutrition (glycemic index, balanced meals), regular physical exercise, and understanding standard lab tests (HbA1c, fasting glucose).
3. SAFETY GUARDRAIL: Never formulate a diagnosis, prescribe medication, or adjust pharmaceutical dosages (e.g., insulin or metformin). Always recommend consulting a licensed medical professional for personal clinical management.
4. Keep answers concise, structured, and easy to read using clean bullet points.
"""

        full_prompt = (
            f"{system_instruction}\n\n"
            f"Recent Conversation:\n{formatted_history}\n\n"
            f"User: {user_query}\n"
            f"Assistant:"
        )

        reply_text = generate_with_resilience(full_prompt)

        # Resilient fallback so the frontend never receives a 500 error during cloud outages
        if not reply_text:
            reply_text = (
                "Our AI assistant is momentarily experiencing high server traffic. "
                "In general, maintaining a balanced diet rich in fiber, staying physically active "
                "with at least 150 minutes of moderate exercise per week, and scheduling regular "
                "HbA1c or fasting glucose screenings are key steps for metabolic wellness. "
                "Please ask your question again in a few moments."
            )

        return ChatResponse(reply=reply_text)

    except Exception as e:
        print("[ERROR] Exception in /chat:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
