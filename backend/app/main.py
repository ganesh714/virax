# /app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import your skill and core routers
from .skills import google_tasks, github_manager, web_search_manager
from .core import chatbot

load_dotenv()

app = FastAPI(
    title="Virax Jr. - AI Agent Hub",
    description="A modular, skill-based AI agent backend.",
    version="0.6.0"
)

# --- CORRECTED MIDDLEWARE SECTION ---
# This is the full, correct code that was missing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Routers ---
# Skills
app.include_router(google_tasks.router, prefix="/api/v1")
app.include_router(github_manager.router, prefix="/api/v1")
app.include_router(web_search_manager.router, prefix="/api/v1")
# Core
app.include_router(chatbot.router, prefix="/api/v1")

# --- Root Endpoint ---
@app.get("/", tags=["Status"])
async def read_root():
    return {"status": "Virax Jr. Agent Hub is online."}