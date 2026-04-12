# ⬡ Bank Churn Intelligence Platform

> Predict customer churn · Explain with AI · Deliver voice insights

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![LLaMA3](https://img.shields.io/badge/LLaMA_3-Groq_API-6f42c1?style=flat-square)
![ElevenLabs](https://img.shields.io/badge/Voice-ElevenLabs-000000?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Overview

The **Bank Churn Intelligence Platform** is a production-grade machine learning application that predicts whether a bank customer will churn, explains the result in plain English using **LLaMA 3**, and delivers the insight as natural speech via **ElevenLabs** — making AI genuinely accessible to non-technical business users.

This project demonstrates end-to-end proficiency across machine learning, MLOps, REST API design, LLM integration, and full-stack engineering.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🔮 **Churn Prediction** | XGBoost on 10,000 records · SMOTE balancing · 87%+ AUC-ROC |
| 🤖 **AI Explanation** | LLaMA 3 via Groq API converts predictions to business-friendly language |
| 🔊 **Voice Intelligence** | ElevenLabs TTS delivers insights as natural human-like audio |
| ⚡ **FastAPI Backend** | Scalable REST API with `/predict`, `/chat`, `/tts` endpoints |
| 💎 **Interactive UI** | Real-time risk ring, feature importance bars, AI chatbot interface |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | XGBoost, Scikit-learn, SMOTE (imbalanced-learn) |
| Backend | FastAPI, Uvicorn, Python 3.10+ |
| AI / LLM | LLaMA 3 70B via Groq API |
| Voice | ElevenLabs Text-to-Speech API |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Data Processing | Pandas, NumPy |

---

## 📁 Project Structure

```
bank-churn-intelligence-platform/
├── backend/
│   ├── main.py                 ← FastAPI app & all endpoints
│   ├── requirements.txt        ← Python dependencies
│   └── Churn_Modelling.csv     ← Dataset (10,000 customers)
├── frontend/
│   └── index.html              ← Full web UI
├── notebooks/
│   └── churn_analysis.ipynb    ← EDA & model development
└── README.md
```

---

## 🚀 Getting Started

### 1 · Clone the repository
```bash
git clone https://github.com/vinayak533/bank-churn-intelligence-platform.git
cd bank-churn-intelligence-platform
```

### 2 · Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3 · Configure API keys

```bash
# macOS / Linux
export GROQ_API_KEY=gsk_your_key_here

# Windows
set GROQ_API_KEY=gsk_your_key_here
```

> Get your free Groq key at [console.groq.com](https://console.groq.com)  
> Get your ElevenLabs key at [elevenlabs.io](https://elevenlabs.io) *(optional — for voice)*

### 4 · Start the backend
```bash
uvicorn main:app --reload --port 8000
```
> ✅ The model trains automatically on startup (~10 seconds)  
> ✅ Interactive API docs available at `http://127.0.0.1:8000/docs`

### 5 · Launch the frontend
Open `frontend/index.html` in your browser — no additional server required.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| AUC-ROC | 87%+ |
| Accuracy | 85%+ |
| Validation | 5-Fold Cross Validation |
| Imbalance Handling | SMOTE Oversampling |

**Key business insights surfaced by the model:**
- Older customers carry significantly higher churn risk
- Customers holding only one product are the most at-risk segment
- Zero-balance accounts are strong churn indicators
- Inactive members contribute disproportionately to churn
- Germany shows a notably higher churn rate vs other regions

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service & model health check |
| `POST` | `/predict` | Run churn prediction |
| `POST` | `/chat` | LLaMA 3 plain-English explanation |
| `POST` | `/tts` | ElevenLabs voice generation |
| `GET` | `/docs` | Swagger interactive documentation |

---

## 💡 System Architecture

```
Customer Data Input
        ↓
  FastAPI Backend
        ↓
  XGBoost Model ──→ Churn Probability · Risk Level · Top Factors
        ↓
  LLaMA 3 (Groq) ──→ Plain English Explanation
        ↓
  ElevenLabs ──→ Voice Audio Output
```

---

## 👤 Author

**Vinayak K V** — Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vinayak--kv--ds-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/vinayak-kv-ds)
[![GitHub](https://img.shields.io/badge/GitHub-vinayak533-181717?style=flat-square&logo=github)](https://github.com/vinayak533)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-gold?style=flat-square)](https://vinayak533.github.io/VINAYAK_PORTFOLIO/)
[![Email](https://img.shields.io/badge/Email-vinayakkvjob@gmail.com-D14836?style=flat-square&logo=gmail)](mailto:vinayakkvjob@gmail.com)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
