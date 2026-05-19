# FastAPI inference server: loads model checkpoint at startup, exposes /health and /predict.

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.tokenizer import Tokenizer
from src.model import TinyDecoderLM
from src.evaluate import greedy_generate

ARTIFACTS_DIR = _ROOT / "artifacts"

_state: dict = {"model": None, "tokenizer": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model checkpoint at startup; release state on shutdown.

    A failed load is logged as a warning — the server stays up and /health
    will report model_loaded=false so callers can detect the degraded state.
    """
    try:
        tok = Tokenizer()
        model = TinyDecoderLM(vocab_size=tok.vocab_size)
        model.load_state_dict(
            torch.load(ARTIFACTS_DIR / "model.pt", weights_only=True)
        )
        model.eval()
        _state["model"] = model
        _state["tokenizer"] = tok
        print("Model loaded successfully.")
    except Exception as exc:
        print(f"WARNING: model load failed — {exc}", file=sys.stderr)
    yield
    _state["model"] = None
    _state["tokenizer"] = None


app = FastAPI(title="Math-LLM-PoC", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    equation: Annotated[str, Field(pattern=r"^\d{1,3}[+\-]\d{1,3}=$")]


class PredictResponse(BaseModel):
    equation: str
    predicted_answer: str
    full_output: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model: TinyDecoderLM | None = _state["model"]
    tokenizer: Tokenizer | None = _state["tokenizer"]

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    prefix_ids = tokenizer.encode(request.equation, add_bos=True, add_eos=False)
    with torch.no_grad():
        full_ids = greedy_generate(model, tokenizer, prefix_ids, max_new_tokens=8)

    latency_ms = (time.perf_counter() - t0) * 1000

    gen_ids = full_ids[len(prefix_ids):]
    if gen_ids and gen_ids[-1] == tokenizer.eos_id:
        gen_ids = gen_ids[:-1]

    return PredictResponse(
        equation=request.equation,
        predicted_answer=tokenizer.decode(gen_ids),
        full_output=tokenizer.decode(full_ids),
        latency_ms=round(latency_ms, 3),
    )


# Entry point: uvicorn src.api:app --host 0.0.0.0 --port 8000
