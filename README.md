# ⬡ BANK CHURN INTELLIGENCE PLATFORM
# Predict Customer Churn · Explain with AI · Deliver Voice Insights

# ============================================================
# 📌 OVERVIEW
# ============================================================
# The Bank Churn Intelligence Platform is a production-grade
# machine learning application designed to:
# - Predict customer churn probability
# - Generate explainable AI insights
# - Deliver voice-based outputs for business users
#
# This project demonstrates end-to-end capabilities in:
# Machine Learning · MLOps · API Development · LLM Integration · Full-Stack Engineering

# ============================================================
# 🎯 KEY FEATURES
# ============================================================
# ✔ Churn Prediction Engine
#   - Built with XGBoost on 10,000 customer records
#   - SMOTE used for class imbalance handling
#   - Achieves 87%+ AUC-ROC

# ✔ Explainable AI (LLM Integration)
#   - LLaMA 3 via Groq API
#   - Converts predictions into simple business-friendly explanations

# ✔ Voice Intelligence Layer
#   - ElevenLabs Text-to-Speech API
#   - Converts insights into natural, human-like voice

# ✔ Scalable Backend
#   - FastAPI REST architecture
#   - Endpoints: /predict, /chat, /tts

# ✔ Interactive Frontend
#   - Real-time predictions
#   - Risk visualization (probability ring, feature importance)
#   - Clean and modern UI

# ============================================================
# 🛠 TECH STACK
# ============================================================
# Machine Learning : XGBoost, Scikit-learn, SMOTE
# Backend          : FastAPI, Uvicorn, Python 3.10+
# AI Model         : LLaMA 3 (Groq API)
# Voice            : ElevenLabs API
# Frontend         : HTML, CSS, JavaScript
# Data Processing  : Pandas, NumPy

# ============================================================
# 📁 PROJECT STRUCTURE
# ============================================================
bank-churn-intelligence-platform/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── Churn_Modelling.csv
├── frontend/
│   └── index.html
├── notebooks/
│   └── churn_analysis.ipynb
└── README.md

# ============================================================
# 🚀 SETUP & INSTALLATION
# ============================================================

# 1. Clone Repository
git clone https://github.com/vinayak533/bank-churn-intelligence-platform.git
cd bank-churn-intelligence-platform

# 2. Install Dependencies
cd backend
pip install -r requirements.txt

# 3. Configure API Keys
# Linux / Mac
export GROQ_API_KEY=gsk_your_key_here

# Windows
set GROQ_API_KEY=gsk_your_key_here

# 4. Run Backend Server
uvicorn main:app --reload --port 8000

# Note:
# Model automatically trains on startup (~10 seconds)

# 5. Launch Frontend
# Open frontend/index.html in your browser

# ============================================================
# 📊 MODEL PERFORMANCE
# ============================================================
# AUC-ROC   : 87%+
# Accuracy  : 85%+
# Validation: 5-Fold Cross Validation
#
# Key Business Insights:
# - Older customers show higher churn risk
# - Customers with only 1 product are high-risk
# - Zero balance accounts are more likely to churn
# - Inactive members contribute significantly to churn
# - Higher churn observed in Germany region

# ============================================================
# 🔌 API ENDPOINTS
# ============================================================
# GET  /health   → Service health check
# POST /predict  → Churn prediction
# POST /chat     → AI explanation (LLM)
# POST /tts      → Voice generation
# GET  /docs     → Swagger API documentation

# ============================================================
# 💡 SYSTEM WORKFLOW
# ============================================================
# Customer Input
#      ↓
# FastAPI Backend
#      ↓
# XGBoost Model → Prediction + Risk Score + Key Factors
#      ↓
# LLaMA 3 (Groq) → Natural Language Explanation
#      ↓
# ElevenLabs → Voice Output

# ============================================================
# 🙋 AUTHOR
# ============================================================
# Name      : VINAYAK K V
# Email     : vinayakkvjob@gmail.com
# Phone     : +91 9037252890
# LinkedIn  : https://www.linkedin.com/in/vinayak-kv-ds
# GitHub    : https://github.com/vinayak533
# Portfolio : https://vinayak533.github.io/VINAYAK_PORTFOLIO/

# ============================================================
# 📄 LICENSE
# ============================================================
# This project is licensed under the MIT License
