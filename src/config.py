"""Application configuration and environment settings."""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    """Global system configuration."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Fact Knowledge Layer"
    environment: str = Field(default="development", description="development | test | production")
    data_dir: str = Field(default=os.path.join(BASE_DIR, "data"))
    upload_dir: str = Field(default=os.path.join(BASE_DIR, "data", "uploads"))
    knowledge_store_path: str = Field(default=os.path.join(BASE_DIR, "data", "knowledge_store"))

    # LLM Settings
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")
    groq_rpm_limit: float = Field(default=24.0, alias="GROQ_RPM_LIMIT")
    groq_rpd_limit: int = Field(default=1000, alias="GROQ_RPD_LIMIT")
    groq_min_interval_seconds: float = Field(default=2.5, alias="GROQ_MIN_INTERVAL_SECONDS")

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="nex-agi/nex-n2.5-mini:free", alias="OPENROUTER_MODEL")
    gemini_model: str = Field(default="gemini-flash-lite-latest", alias="GEMINI_MODEL")
    default_llm_model: str = Field(default="openai/gpt-oss-120b", alias="DEFAULT_LLM_MODEL")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")


settings = Settings()
