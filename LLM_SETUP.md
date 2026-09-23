# Woodland OCR — OpenRouter Setup

Woodland OCR uses [OpenRouter](https://openrouter.ai) for both the document/adjustment
extraction (`/extract-transaction`) and the chat/agent endpoints (`/agent/chat`,
`/agent/chat/stream`) in `agent_brain.py`. One key, one provider.

## Setup (2 minutes)

1. **Get an OpenRouter API key**: https://openrouter.ai/keys
2. **Set it as an environment variable** (see `.env` at the repo root, loaded by
   `start-dev.ps1`):
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-actual-key
   OPENROUTER_MODEL=openai/gpt-4o-mini
   OPENROUTER_EXTRACT_MODELS=openai/gpt-4o-mini,anthropic/claude-3.5-haiku,meta-llama/llama-3.1-8b-instruct
   ```
3. **Restart the app** (or just start it via `start-dev.ps1`, which now launches this
   service automatically):
   ```bash
   python app.py
   ```

## How It Works

1. **OpenRouter** — tries each model in `OPENROUTER_EXTRACT_MODELS`, in order, to
   extract structured transaction/adjustment data (contractual rent, management
   fees, building expenditure, VAT, etc. — the same fields the Landlord Payment
   adjustment breakdown uses).
2. **Pattern matching fallback** — if no key is configured or every model fails,
   falls back to regex extraction so the endpoint always returns something usable.

## Testing

```bash
curl -X POST "http://localhost:5006/extract-transaction" -F "file=@your_invoice.png"
```

## Troubleshooting

- **"Invalid API key"**: check `OPENROUTER_API_KEY` in your `.env`.
- **Still getting fallback**: check the app logs — a listed model may be
  unavailable on your OpenRouter plan; trim `OPENROUTER_EXTRACT_MODELS` to
  models you have access to.
