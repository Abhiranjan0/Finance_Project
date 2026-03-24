# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import (
    chat,
    risk_profile,
    simulate_portfolio,
    rag,
    nav,
    currency,
    report,
    portfolio,   
    simulation,      
    debug,
    conversation, 
    auth
)


app = FastAPI(
    title="Finance Advisor Backend",
    description="AI-powered financial advisor using Azure OpenAI.",
    version="1.0.0"
)

# -----------------------------
# CORS SETTINGS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Change to your Streamlit domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# ROUTER REGISTRATION
# -----------------------------
app.include_router(chat.router)
app.include_router(risk_profile.router)
app.include_router(simulate_portfolio.router)
app.include_router(rag.router)
app.include_router(nav.router)
app.include_router(currency.router)
app.include_router(report.router)
app.include_router(portfolio.router)
app.include_router(simulation.router)
app.include_router(debug.router)
app.include_router(conversation.router)
app.include_router(auth.router)



# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "azure_endpoint": settings.groq_api_key,
        "model": settings.groq_model
    }
