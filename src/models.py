from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    thread_id: str = Field(..., min_length=3, max_length=120)

class SourceItem(BaseModel):
    type: Literal["internal", "web"]
    title: str = ""
    source: str = ""
    url: Optional[str] = None
    page: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    route: str
    used_web_search: bool
    support_status: str = ""
    usefulness: str = ""
    sources: List[SourceItem] = []
    trace: List[str] = []
    thread_id: str
    memory_turns: int = 0
    request_id: str = ""

class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    namespace: str


# --------------------------------------------------------------------------- #
# Live evaluation dashboard schemas
# --------------------------------------------------------------------------- #
class EvaluationSummaryResponse(BaseModel):
    """Overall summary. Nested metric blocks carry their own ``status``."""

    date_from: Optional[str] = None
    date_to: Optional[str] = None
    requests: Dict[str, Any]
    latency: Dict[str, Any]
    self_rag: Dict[str, Any]
    tokens: Dict[str, Any]
    quality: Dict[str, Any]
    legend: Dict[str, str] = {}


class EvaluationComparisonResponse(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    systems: Dict[str, Any]
    note: str = ""


class DailyMetricsPoint(BaseModel):
    date: str
    requests: int = 0
    success: int = 0
    failed: int = 0
    success_rate: Optional[float] = None
    failure_rate: Optional[float] = None
    evaluations: int = 0
    latency: Dict[str, Any]
    status: str = "ok"


class DailyMetricsResponse(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    points: List[DailyMetricsPoint] = []
    status: str = "ok"


class HistoryItem(BaseModel):
    request_id: str
    created_at: str
    question: str
    status: str
    latency_seconds: Optional[float] = None
    error: Optional[str] = None
    error_category: Optional[str] = None
    self_rag: Optional[Dict[str, Any]] = None
    baseline: Optional[Dict[str, Any]] = None
    evaluation: Dict[str, Any] = {}


class EvaluationHistoryResponse(BaseModel):
    page: int = 1
    page_size: int = 20
    total: int = 0
    total_pages: int = 0
    has_next: bool = False
    has_prev: bool = False
    items: List[HistoryItem] = []
    status: str = "ok"


# --------------------------------------------------------------------------- #
# Optional evaluator schemas
# --------------------------------------------------------------------------- #
class JudgeResponse(BaseModel):
    request_id: str
    system_version: str
    status: str
    score: Optional[Dict[str, Any]] = None


class GradeBaselineResponse(BaseModel):
    request_id: str
    system_version: str
    status: str
    grades: List[Dict[str, Any]] = []

