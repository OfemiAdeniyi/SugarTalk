"""
SugarTalk Backend
==================
Exposes:
  1. POST /predict -> ML screening probability + risk tier + Gemini recommendations
  2. POST /chat    -> Interactive diabetes Q&A assistant with clinical guardrails
  3. GET  /health  -> Health and service readiness check

Environment variables expected:
  GEMINI_API_KEY -> Google GenAI API key from aistudio.google.com/apikey
"""

import os
import pickle
import traceback
from typing import Any, List, Literal, Optional

import pandas as pd
from google import genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="SugarTalk Diabetes Screening & Q&A API")

# --- CORS Middleware ---
# Configured to allow cross-origin requests from Lovable and local web clients
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
PRIMARY_LLM = "gemini-2.5-flash"
FALLBACK_LLM = "gemini-2.0-flash"


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
    text: Optional[str] = None  # Fallback field for Lovable frontend variants
    risk_tier: Optional[str] = "Unknown"
    history: Optional[List[Any]] = []  # Permissive list to handle varying dict/object structures


class ChatResponse(BaseModel):
    reply: str


# --- Helper Functions ---
def risk_tier_from_probability(probability: float, threshold: float) -> str:
    """Classifies risk into display buckets."""
    if probability < 0.15:
        return "Low"
    elif probability < threshold:
        return "Moderate"
    else:
        return "High"


def build_recommendation_prompt(data: ScreeningInput, probability: float, tier: str) -> str:
    return f"""A user completed a diabetes risk screening (NOT a formal clinical diagnosis) with these inputs:
- Gender: {data.gender}
- Age: {data.age}
- Hypertension: {"Yes" if data.hypertension else "No"}
- Heart disease: {"Yes" if data.heart_disease else "No"}
- Smoking history: {data.smoking_history}
- BMI: {data.bmi}

Model output: estimated risk probability {probability:.0%}, risk tier: {tier}.

Write a short, warm, supportive, and non-alarming message for the user with:
1. One plain-language sentence explaining what this risk tier indicates.
2. 3-4 practical lifestyle tips relevant to their inputs (nutrition, activity, smoking cessation, weight management).
3. A clear recommendation for a confirmatory fasting glucose or HbA1c lab test at a clinic (make this firmer if Moderate or High).

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
        # Construct DataFrame matching the scikit-learn pipeline columns
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

        recommendations = (
            "We recommend consulting a healthcare professional for standard confirmatory "
            "laboratory testing (such as a Fasting Blood Glucose or HbA1c test)."
        )

        if gemini_client:
            try:
                prompt = build_recommendation_prompt(data, probability, tier)
                try:
                    response = gemini_client.models.generate_content(
                        model=PRIMARY_LLM,
                        contents=prompt,
                    )
                except Exception as primary_err:
                    print(f"[WARN] Primary model failed in /predict: {primary_err}. Trying fallback...")
                    response = gemini_client.models.generate_content(
                        model=FALLBACK_LLM,
                        contents=prompt,
                    )

                if response and response.text:
                    recommendations = response.text
            except Exception as llm_err:
                print(f"[ERROR] LLM generation failed in /predict: {llm_err}")

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

    if not gemini_client:
        print("[ERROR] gemini_client is None. Ensure GEMINI_API_KEY is set in Render Environment variables.")
        raise HTTPException(
            status_code=500,
            detail="AI assistant service is currently unconfigured on the server."
        )

    try:
        # Build conversational history safely regardless of frontend payload formatting
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

        response_text = ""
        try:
            resp = gemini_client.models.generate_content(
                model=PRIMARY_LLM,
                contents=full_prompt,
            )
            response_text = resp.text
        except Exception as primary_err:
            print(f"[WARN] Primary model failed in /chat ({primary_err}). Trying fallback {FALLBACK_LLM}...")
            resp = gemini_client.models.generate_content(
                model=FALLBACK_LLM,
                contents=full_prompt,
            )
            response_text = resp.text

        return ChatResponse(
            reply=response_text or "I'm here to help, but could not generate a response. Please rephrase your question."
        )

    except Exception as e:
        print("[ERROR] Exception in /chat:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
