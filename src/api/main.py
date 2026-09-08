"""FastAPI application entrypoint for Fact Knowledge Layer."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.config import settings

app = FastAPI(
    title="Fact Knowledge Layer API",
    description="Cross-Document Fact Knowledge Layer with Contextual Reconciliation (Superjoin VIT 2026 Assignment)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Welcome to Fact Knowledge Layer API",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
    }
