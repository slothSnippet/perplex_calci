# Perplexity Explorer

A small web app that scores a sentence with the `distilgpt2` language model and shows each token's probability, sentence perplexity, and likely next tokens.

## Run locally

Use Python 3.10 or newer. From this project directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000>. The model weights are downloaded from Hugging Face the first time you calculate a sentence and cached for later requests. Set `MODEL_ID` to another compatible causal language model before starting the server if desired.

## API

- `GET /api/health` reports the service and configured model.
- `POST /api/analyze` accepts `{"text":"I love Python"}` and returns per-token probabilities, perplexity, and the top next-token predictions.

The sentence is scored with the model's ordinary next-token probabilities, using its end-of-text token as the initial context so the first visible token is scored too. The hand-entered examples remain entirely in the browser and are labeled as simulated.
