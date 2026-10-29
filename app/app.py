"""FastAPI app: upload a resume, get matched jobs with explanations."""
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pypdf.errors import PdfReadError

import pipeline
from llm import LLMLimitError

APP_DIR = Path(__file__).parent
MAX_BYTES = 2 * 1024 * 1024          # 2 MB upload limit
ALLOWED = (".pdf", ".txt")

SAMPLES_FILE = APP_DIR / "data" / "samples.json"
samples = json.loads(SAMPLES_FILE.read_text(encoding="utf-8")) if SAMPLES_FILE.exists() else []

app = FastAPI(title="Resume Job Matcher")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/samples")
def list_samples():
    return [{"id": i, "label": s["label"]} for i, s in enumerate(samples)]


@app.get("/api/samples/{sample_id}")
def get_sample(sample_id: int):
    if not 0 <= sample_id < len(samples):
        raise HTTPException(404, "Sample not found.")
    return samples[sample_id]["result"]


@app.post("/api/match")
def match(file: UploadFile = File(...)):
    name = file.filename or ""
    if not name.lower().endswith(ALLOWED):
        raise HTTPException(400, "Please upload a PDF or .txt file.")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(400, "File too large (max 2 MB).")
    try:
        return pipeline.match(data, name)        # resume is processed in memory, never saved
    except (ValueError, PdfReadError) as e:
        raise HTTPException(422, str(e) or "Could not read this file.")
    except LLMLimitError as e:
        raise HTTPException(429, str(e))
