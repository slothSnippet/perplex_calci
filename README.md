# Perplexity Explorer

This is a static website. The prompt model runs in the visitor's browser, so deployment does not require a Python server, API, build step, or environment variables.

## Deploy on Vercel

Import the repository into Vercel and deploy. The included `vercel.json` selects the static “Other” preset and clears build and install commands, so the project does not need dashboard build settings. Keep the project at the repository root; the site is served from `index.html`.

The first calculation downloads the quantized DistilGPT-2 model from Hugging Face through Transformers.js. Later calculations reuse the model in that browser session. Visitors need an internet connection for the initial model download.
