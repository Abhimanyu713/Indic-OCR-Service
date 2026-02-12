import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4o"  # Recommended for high-quality multilingual OCR
    max_tokens: int = 4096

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()