"""Application configuration, loaded from environment variables."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()


@dataclass(frozen=True)
class Settings:
    github_webhook_secret: str
    # Postgres + pgvector DSN for the RAG index. Optional: when unset, the app
    # runs exactly as before (diff-only reviews) and RAG is not wired up.
    database_url: str | None = None

    @property
    def rag_enabled(self) -> bool:
        return bool(self.database_url)


def get_settings() -> Settings:
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    if not secret:
        raise RuntimeError(
            "GITHUB_WEBHOOK_SECRET is not set. Copy .env.example to .env and fill it in."
        )
    return Settings(
        github_webhook_secret=secret,
        database_url=os.getenv("DATABASE_URL") or None,
    )
