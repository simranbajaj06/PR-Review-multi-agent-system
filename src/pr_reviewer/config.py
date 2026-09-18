"""Application configuration, loaded from environment variables."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Settings:
    github_webhook_secret: str


def get_settings() -> Settings:
    
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    if not secret:
        raise RuntimeError(
            "GITHUB_WEBHOOK_SECRET is not set. Copy .env.example to .env and fill it in."
        )
    return Settings(github_webhook_secret=secret)
