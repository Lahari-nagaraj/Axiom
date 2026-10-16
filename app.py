from pathlib import Path
import shutil
import time
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from src.models import ChatRequest, ChatResponse, UploadResponse
from src.self_rag import run_self_rag
from src.ingestion import ingest_file, namespace, SUPPORTED
from src.db import init_db, save_audit, latest_audits
from src.live_db import init_live_db
from src.live_eval import new_request_id, record_chat, schedule_baseline_shadow
from src.api_eval import router as evaluation_router
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)


app = FastAPI(
    title="Axiom — Enterprise Incident Response Self-RAG Copilot",
    version="2.0.0",
    description="Self-RAG copilot for cloud operations, production troubleshooting, and incident-response runbooks.",
)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.include_router(evaluation_router)



@app.on_event("startup")
def startup():
    init_db()
    init_live_db()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "cloudops-sentinel-self-rag"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, background_tasks: BackgroundTasks):
    question = payload.question.strip()
    thread_id = payload.thread_id.strip()
    request_id = new_request_id()
    started = time.perf_counter()
    try:
        result = await run_in_threadpool(run_self_rag, question, thread_id)
    except Exception as e:
        latency = time.perf_counter() - started
        # Logging the failure must never mask the original error.
        try:
            await run_in_threadpool(record_chat, request_id, question, thread_id, None, latency, e)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.perf_counter() - started
    # Persist the request + Self-RAG evaluation. record_chat is fail-safe.
    try:
        await run_in_threadpool(record_chat, request_id, question, thread_id, result, latency)
    except Exception:
        pass

    # Legacy audit table kept for backwards compatibility; a failure is non-fatal.
    try:
        await run_in_threadpool(save_audit, question, result)
    except Exception:
        pass

    # Baseline RAG shadow evaluation runs after the response is sent.
    try:
        schedule_baseline_shadow(background_tasks, request_id, question)
    except Exception:
        pass
    return ChatResponse(request_id=request_id, **result)


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail="Supported: PDF, TXT, MD, DOCX")
    safe_name = Path(file.filename).name
    target = UPLOADS / safe_name
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        count = await run_in_threadpool(ingest_file, target)
        return UploadResponse(filename=safe_name, chunks_indexed=count, namespace=namespace())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audits")
def audits(limit: int = 20):
    return latest_audits(min(max(limit, 1), 100))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)




