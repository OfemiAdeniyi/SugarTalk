# 🩸 SugarTalk

**Clinical-Grade Diabetes Risk Screening & Interactive AI Wellness Companion**

🔗 **Live App:** [sugar-insight-ai.lovable.app](https://sugar-insight-ai.lovable.app)

SugarTalk is a preventive digital health application that bridges predictive machine learning with conversational generative AI. It screens individuals for diabetes risk using non-invasive, self-reported biometric indicators and translates raw statistical probabilities into empathetic, personalized lifestyle guidance and clinical next steps.

---

## 🌟 Overview

Early detection of metabolic disease remains a critical hurdle in public health. SugarTalk combines:

1. **A Scikit-Learn Random Forest Classifier** — Evaluates risk probability based on demographic and lifestyle indicators (Age, BMI, Hypertension, Heart Disease, Smoking History).
2. **Recall-Tuned Clinical Thresholding** — Prioritizes screening sensitivity to minimize false negatives before symptoms escalate.
3. **Gemini Generative AI** — Translates numerical probabilities into actionable, warm, non-diagnostic lifestyle recommendations and powers a contextual Q&A diabetes educator assistant.
4. **Care-Access & Monetization Handoff** — Connects screened users directly to partnered diagnostic laboratories (HbA1c/Fasting Glucose) and virtual teleconsultations with endocrinologists.

---

## 🏗️ System Architecture

```text
[User / Browser]
       │
       ▼
[Lovable Frontend (React + Tailwind CSS + shadcn/ui)]
       │
       │  POST /predict (Biometrics)  OR  POST /chat (User Inquiry)
       ▼
[Render Web Service: FastAPI Backend]
       ├── diabetes_screening_model.pkl (Scikit-Learn Pipeline)
       ├── screening_threshold.pkl     (Recall-optimized Decision Boundary)
       └── Google GenAI SDK (Multi-model resilient fallback pool)
               │
               ▼
[Structured Response: Risk Tier + Clinical Alert + Action Plan]
```

**Frontend:** [sugar-insight-ai.lovable.app](https://sugar-insight-ai.lovable.app)

---

## 🚀 Key Features

- **Instant Risk Stratification** — Computes continuous probability and assigns discrete tiers: Low, Moderate, or High Risk.
- **Clinical Safety Guardrails** — Hard-coded prompt protections ensure the assistant provides lifestyle education without prescribing medication, altering dosages (e.g., insulin/metformin), or issuing formal diagnoses.
- **Resilient Multi-Model Fallback** — Production-hardened against cloud capacity spikes (503 UNAVAILABLE / 429 Rate Limits) with exponential backoff and automatic model failover.
- **Context-Aware AI Chat** — Users can ask clarifying questions about lab tests, glycemic indexes, and meal planning, with their screening context automatically retained.
- **Clinical Next-Steps Integration** — Embedded modules for booking confirmatory laboratory panels and scheduling specialist teleconsultations.

---

## 📁 Repository Structure

```text
.
├── main.py                          # FastAPI application (predict, chat, health endpoints)
├── diabetes_screening_model.pkl     # Serialized Scikit-Learn Random Forest model
├── screening_threshold.pkl          # Recall-tuned decision boundary
├── requirements.txt                 # Backend Python dependencies
├── .gitignore                       # Ignored local files, environments, and secrets
└── README.md                        # Documentation
```

---

## 🛠️ API Reference

### 1. Health Check

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "ok",
  "service": "SugarTalk API",
  "gemini_connected": true
}
```

### 2. Risk Prediction & Screening

**Endpoint:** `POST /predict`

**Request Body:**

```json
{
  "gender": "Female",
  "age": 48.0,
  "hypertension": 1,
  "heart_disease": 0,
  "smoking_history": "former",
  "bmi": 28.4
}
```

**Response:**

```json
{
  "probability": 0.3821,
  "risk_tier": "Moderate",
  "flagged": true,
  "recommendations": "### Clinical Overview\nYour inputs indicate a moderate risk tier..."
}
```

### 3. Diabetes Q&A Assistant

**Endpoint:** `POST /chat`

**Request Body:**

```json
{
  "message": "What are low glycemic index alternatives to white rice?",
  "risk_tier": "Moderate",
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I assist your metabolic health journey today?"}
  ]
}
```

**Response:**

```json
{
  "reply": "Here are several nutritious, lower-glycemic alternatives..."
}
```

---

## ⚙️ Local Development Setup

### Prerequisites

- Python 3.10+
- Google AI Studio API Key ([Get an API Key](https://aistudio.google.com/))

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/<your-username>/sugartalk-backend.git
cd sugartalk-backend
```

**2. Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Set environment variables:**

```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
# On Windows (PowerShell): $env:GEMINI_API_KEY="your-gemini-api-key-here"
```

**5. Start the development server:**

```bash
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` to test endpoints via the interactive Swagger UI.

---

## 🚢 Deployment (Render)

1. Create a **New Web Service** on Render.
2. Connect this GitHub repository.
3. Configure settings:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
4. Under **Environment Variables**, add:
   - `GEMINI_API_KEY` = `<your-api-key>`
5. Deploy and copy your generated public backend URL.

---

## 🛡️ Medical & Ethical Disclaimer

SugarTalk is an informational risk-screening and wellness educational prototype. It does not provide medical diagnoses, clinical treatment plans, or pharmaceutical prescriptions. Users flagged with elevated risk indicators are strictly instructed to seek validation from licensed medical professionals and accredited diagnostic pathology laboratories.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
