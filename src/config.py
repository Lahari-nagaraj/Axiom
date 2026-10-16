from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float_or_none(name: str) -> Optional[float]:
    """Parse an optional float env var. Empty/invalid -> None (never 0)."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


class Settings(BaseModel):
    # Groq
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    # HuggingFace Embeddings
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5"
    )

    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "self-rag")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "cloudops-docs")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # Internet search
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # Self-RAG controls
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_support_retries: int = int(os.getenv("MAX_SUPPORT_RETRIES", "2"))
    max_retrieval_rewrites: int = int(os.getenv("MAX_RETRIEVAL_REWRITES", "2"))
    max_web_rewrites: int = int(os.getenv("MAX_WEB_REWRITES", "2"))
    database_path: str = os.getenv("DATABASE_PATH", "data/audit.db")

    # Live evaluation (runtime request logging + shadow Baseline RAG)
    live_eval_enabled: bool = _env_bool("LIVE_EVAL_ENABLED", True)
    live_eval_baseline_enabled: bool = _env_bool("LIVE_EVAL_BASELINE_ENABLED", True)
    live_eval_max_concurrency: int = int(os.getenv("LIVE_EVAL_MAX_CONCURRENCY", "2"))
    # Optional LLM evaluator (off by default). When enabled AND a provider key is
    # configured, the dashboard can run a bounded LLM-as-judge score on demand.
    # The deterministic automatic evaluation never needs it.
    live_eval_judge_enabled: bool = _env_bool("LIVE_EVAL_JUDGE_ENABLED", False)

    # Optional explicit Groq pricing (USD per 1M tokens). Unset -> cost is
    # reported as unavailable rather than invented.
    groq_input_cost_per_1m: Optional[float] = _env_float_or_none("GROQ_INPUT_COST_PER_1M")
    groq_output_cost_per_1m: Optional[float] = _env_float_or_none("GROQ_OUTPUT_COST_PER_1M")

    @property
    def database_file(self) -> Path:
        p = Path(self.database_path)
        return p if p.is_absolute() else ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()