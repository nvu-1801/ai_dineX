from dotenv import load_dotenv
import os

load_dotenv()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    GEMINI_API_KEY: str
    DATABASE_URL: str
    MAIN_BACKEND_URL: str = ""
    INTERNAL_API_KEY: str = "dinex-rag-internal-key-8f9a2b"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
