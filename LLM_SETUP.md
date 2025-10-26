# Groq Free API Setup Guide

This system uses Groq's free API for fast and reliable transaction data extraction.

## Setup (2 minutes)

1. **Get Groq API Key**:
   - Go to [Groq Console](https://console.groq.com/keys)
   - Sign up for a free account (no credit card required)
   - Create an API key

2. **Update Configuration**:
   - Open `llm_config.py`
   - Replace `gsk_your_token_here` with your actual token:
   ```python
   GROQ_TOKEN = "gsk_your_actual_token_here"
   ```

3. **Restart Your App**:
   ```bash
   python3 app.py
   ```

## Free Tier Limits

- **14,400 requests per day** (plenty for most use cases)
- **Fast responses** (usually under 1 second)
- **No credit card required**

## How It Works

1. **Groq API** - Tries to extract structured data using AI
2. **Pattern Matching** - Falls back to regex extraction if Groq fails
3. **Always works** - Guaranteed to return transaction data

## Testing

Test the setup:
```bash
curl -X POST "http://localhost:5006/extract-transaction" \
  -F "file=@your_invoice.png"
```

## Troubleshooting

- **"Invalid API key"**: Check your token in `llm_config.py`
- **"Rate limit exceeded"**: Wait a few minutes (14,400 requests/day is generous)
- **Still getting fallback**: Check logs to see what's happening

The system will always work - if Groq fails, it uses pattern matching as backup!
