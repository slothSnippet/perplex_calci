"""Small FastAPI service for GPT-2 prompt perplexity and next-token predictions."""

from functools import lru_cache
import math
import os

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.getenv("MODEL_ID", "distilgpt2")
MAX_INPUT_CHARS = 20_000
TOP_K = 8

app = FastAPI(title="Perplexity Explorer API", version="1.0.0")


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


@lru_cache(maxsize=1)
def get_model():
    """Load the tokenizer and model once, on the first analysis request."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter a sentence before calculating.")

    try:
        tokenizer, model = get_model()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise HTTPException(status_code=422, detail="The sentence did not produce any tokens.")

        # GPT-2 has no separate BOS token. Its EOS token is used as a start marker
        # so the first sentence token can also be scored.
        input_ids = [tokenizer.eos_token_id, *token_ids]
        max_positions = getattr(model.config, "n_positions", 1024)
        if len(input_ids) > max_positions:
            raise HTTPException(
                status_code=422,
                detail=f"This sentence is too long for {MODEL_ID} ({len(token_ids)} tokens; maximum is {max_positions - 1}).",
            )

        inputs = torch.tensor([input_ids], dtype=torch.long)
        with torch.inference_mode():
            logits = model(input_ids=inputs).logits[0]
            # logits[i] predicts token_ids[i], because input position zero is the start marker.
            scored_logits = logits[: len(token_ids)].float()
            log_probs = torch.log_softmax(scored_logits, dim=-1)
            selected = log_probs[torch.arange(len(token_ids)), torch.tensor(token_ids)]
            probabilities = selected.exp().tolist()
            perplexity = math.exp(-sum(selected.tolist()) / len(token_ids))

            next_probs = torch.softmax(logits[-1].float(), dim=-1)
            top_probs, top_ids = torch.topk(next_probs, k=TOP_K)

        tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
        next_tokens = [
            {"token": tokenizer.decode([int(token_id)]) or "(special token)", "probability": float(probability)}
            for token_id, probability in zip(top_ids.tolist(), top_probs.tolist())
        ]
        return {
            "model": MODEL_ID,
            "tokens": [
                {"token": token, "probability": probability}
                for token, probability in zip(tokens, probabilities)
            ],
            "perplexity": perplexity,
            "next_tokens": next_tokens,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Could not load or run the language model: {error}") from error


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "index.html"))


@app.get("/{asset_path:path}")
def frontend_fallback(asset_path: str):
    # Serve the single-page frontend for client-side navigation and common root requests.
    if asset_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "index.html"))
