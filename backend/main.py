from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle, io, os, httpx
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── Load environment variables from .env ──────────────────────
load_dotenv()

app = FastAPI(title="Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────
model = None
scaler = None
feature_names = None

# ── Train on startup ──────────────────────────────────────────
@app.on_event("startup")
def train_model():
    global model, scaler, feature_names

    df = pd.read_csv("Churn_Modelling.csv")
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    X = df.drop(columns=["Exited"])
    y = df["Exited"]
    feature_names = list(X.columns)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    with open("churn_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("[OK] Model trained and saved!")


# ── Request schemas ────────────────────────────────────────────
class CustomerInput(BaseModel):
    CreditScore: float
    Gender: int           # 1=Male 0=Female
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_Germany: int
    Geography_Spain: int

class ChatRequest(BaseModel):
    message: str
    prediction_context: dict

class TTSRequest(BaseModel):
    text: str


# ── Prediction endpoint ────────────────────────────────────────
@app.post("/predict")
def predict(customer: CustomerInput):
    data = pd.DataFrame([customer.model_dump()])
    data_scaled = scaler.transform(data)

    prob       = float(model.predict_proba(data_scaled)[0][1])
    prediction = int(model.predict(data_scaled)[0])

    importances = model.feature_importances_
    top_factors = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1], reverse=True
    )[:5]
    top_factors = [{"feature": k, "importance": round(float(v), 4)} for k, v in top_factors]

    risk_level = "HIGH" if prob > 0.65 else "MEDIUM" if prob > 0.35 else "LOW"

    return {
        "probability": round(prob * 100, 2),
        "prediction": prediction,
        "risk_level": risk_level,
        "top_factors": top_factors,
        "customer_data": customer.model_dump()
    }


# ── Groq Chatbot endpoint ──────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY not set")

    ctx = req.prediction_context
    system_prompt = f"""You are a friendly banking AI advisor named ARIA.
A customer churn prediction was just made. Here are the results:
- Churn Probability: {ctx.get('probability', 'N/A')}%
- Risk Level: {ctx.get('risk_level', 'N/A')}
- Prediction: {'Will CHURN' if ctx.get('prediction') == 1 else 'Will STAY'}
- Top Factors: {ctx.get('top_factors', [])}
- Customer Data: {ctx.get('customer_data', {})}

Explain this prediction in simple, friendly language a non-technical bank customer can understand.
Be empathetic, concise (3-4 sentences), and give 1-2 actionable recommendations.
Never use technical jargon like 'model', 'algorithm', 'features', or 'SHAP'.
Speak directly to the bank staff member reviewing this customer.
"""

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": req.message}
                ],
                "max_tokens": 400,
                "temperature": 0.7
            },
            timeout=30
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return {"reply": resp.json()["choices"][0]["message"]["content"]}


# ── ElevenLabs TTS endpoint ────────────────────────────────────
@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    voice_id = "EXAVITQu4vr4xnSDxMaL"   # Sarah — warm, professional
    eleven_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not eleven_key:
        raise HTTPException(status_code=400, detail="ELEVENLABS_API_KEY not set")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": eleven_key,
                "Content-Type": "application/json"
            },
            json={
                "text": req.text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
            },
            timeout=30
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="ElevenLabs TTS failed")

    return StreamingResponse(io.BytesIO(resp.content), media_type="audio/mpeg")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
