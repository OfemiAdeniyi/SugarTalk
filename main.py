"""
SugarTalk Backend
==================
Exposes:
  1. POST /predict -> ML screening probability + risk tier + Gemini recommendations
  2. POST /chat    -> Interactive diabetes Q&A assistant with clinical safety guardrails
  3. GET  /health  -> Status check
"""

import os
import pickle
from typing import List, Literal, Optional

import pandas as pd
from google import genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="SugarTalk Diabetes Screening & Q&A API")

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
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=api_key) if api_key else None
LLM_MODEL = "gemini-2.5-flash"


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
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    risk_tier: Optional[str] = "Unknown"  # Optional context from their latest screening
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    reply: str


# --- Helpers ---
def risk_tier_from_probability(probability: float, threshold: float) -> str:
    if probability < 0.15:
        return "Low"
    elif probability < threshold:
        return "Moderate"
    else:
        return "High"


def build_recommendation_prompt(data: ScreeningInput, probability: float, tier: str) -> str:
    return f"""A user completed a diabetes risk screening (NOT a diagnosis) with these details:
- Gender: {data.gender}
- Age: {data.age}
- Hypertension: {"Yes" if data.hypertension else "No"}
- Heart disease: {"Yes" if data.heart_disease else "No"}
- Smoking history: {data.smoking_history}
- BMI: {data.bmi}

Model output: estimated risk probability {probability:.0%}, risk tier: {tier}.

Write a short, warm, non-alarming message for the user with:
1. One plain-language sentence on what this risk tier means.
2. 3-4 practical lifestyle tips relevant to their specific inputs.
3. A clear line recommending a confirmatory HbA1c or fasting glucose test at a clinic.

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

        recommendations = "Consult a licensed healthcare provider for standard laboratory testing."
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
                print(f"Gemini error: {llm_err}")

        return ScreeningResult(
            probability=round(probability, 4),
            risk_tier=tier,
            flagged=flagged,
            recommendations=recommendations,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(req: ChatRequest):
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini client is not configured on server.")

    try:
        # Build context from previous conversation turns
        formatted_history = "\n".join([f"{m.role.capitalize()}: {m.content}" for m in req.history[-6:]])

        system_instruction = f"""
You are SugarTalk Assistant, a supportive, knowledgeable diabetes educator and wellness guide.

Context:
- The user's current assessed screening risk tier is: {req.risk_tier}.

Instructions:
1. Answer the user's questions clearly in plain, empathetic language.
2. Focus on education, nutrition (glycemic index, meal balance), physical exercise, and understanding lab tests (HbA1c, OGTT, fasting glucose).
3. SAFETY GUARDRAIL: Never diagnose diseases, prescribe medications, or adjust pharmaceutical dosages (like insulin or metformin). Always advise consulting a licensed physician for medical changes.
4. Keep replies concise, structured, and easy to read.
"""

        full_prompt = f"{system_instruction}\n\nRecent Conversation:\n{formatted_history}\nUser: {req.message}\nAssistant:"

        response = gemini_client.models.generate_content(
            model=LLM_MODEL,
            contents=full_prompt,
        )

        return ChatResponse(reply=response.text or "I'm sorry, I couldn't process that question. Please try asking again.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
