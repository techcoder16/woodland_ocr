"""Woodland OCR agent brain: OpenRouter reasoning, local RAG, and MCP-style tools."""
import json
import os
from typing import Any, Dict, Optional

import requests

# The key lives in .env, not the process environment, so load it the same way
# llm_config does — without this every call here raised "not configured".
from llm_config import OPENROUTER_API_KEY, OPENROUTER_MODELS


class WoodlandAgent:
    def __init__(self) -> None:
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    def _api_key(self) -> str:
        """Prefer a live process env var, else the .env value loaded at import."""
        key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        return key

    def _retrieve(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Small deterministic RAG layer; replace with vector retrieval as the corpus grows."""
        documents = (context or {}).get("documents", [])
        terms = set(query.lower().split())
        ranked = sorted(
            documents,
            key=lambda doc: sum(term in str(doc).lower() for term in terms),
            reverse=True,
        )
        return json.dumps(ranked[:8], default=str)

    def _tools(self) -> list[Dict[str, Any]]:
        return [
            {"name": "get_property_context", "description": "Read property and transaction context supplied by Woodland API"},
            {"name": "search_documents", "description": "Search indexed Woodland OCR documents using RAG"},
            {"name": "summarize_transaction", "description": "Explain extracted transaction values and warnings"},
        ]

    def complete(self, prompt: str, context: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> str:
        api_key = self._api_key()
        rag_context = self._retrieve(prompt, context)
        system = (
            "You are Woodland OCR, the property-management AI brain. "
            "Use only supplied backend context and retrieved documents. "
            "Never invent financial values. State when data is missing. "
            f"Available MCP tools: {json.dumps(self._tools())}"
        )
        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://woodlandltd.com"),
                "X-Title": "Woodland OCR",
            },
            json={
                "model": model or self.model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Retrieved context:\n{rag_context}\n\n{prompt}"},
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError("OpenRouter returned an empty response")
        return content.strip()

    def audit_data(self, question: str, summary: Dict[str, Any], records: list, model: Optional[str] = None) -> str:
        """Answer a data-completeness question from a Woodland audit scan.

        `records` carries field names and presence only — never stored values — so
        the model can report what is missing without ever seeing personal data.
        """
        api_key = self._api_key()

        # Rank the gap records against the question so the most relevant ones
        # survive the slice when the corpus is larger than the context budget.
        terms = {term for term in question.lower().split() if len(term) > 2}
        ranked = sorted(
            records,
            key=lambda record: sum(term in str(record).lower() for term in terms),
            reverse=True,
        )[:60]

        system = (
            "You are the Woodland data-completeness auditor. You are given a summary of a "
            "database scan and a list of records with missing fields. Report only what the "
            "data shows: which records are missing which fields, and which gaps are most "
            "common. Never invent records, field names or counts. When asked what data is "
            "present, infer it from completeness percentages and the absence of a field from "
            "the missing list. You see field names only, never stored values, so never claim "
            "to know what a field contains. Be concise and lead with the biggest gaps."
        )
        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://woodlandltd.com"),
                "X-Title": "Woodland Data Audit",
            },
            json={
                "model": model or self.model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            f"Scan summary:\n{json.dumps(summary, default=str)}\n\n"
                            f"Records with missing fields:\n{json.dumps(ranked, default=str)}\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError("OpenRouter returned an empty response")
        return content.strip()

    def stream(self, prompt: str, context: Optional[Dict[str, Any]] = None, model: Optional[str] = None):
        api_key = self._api_key()
        messages = [
            {"role": "system", "content": "You are Woodland OCR, a precise property-management assistant. Use only supplied context and never invent financial values."},
            {"role": "user", "content": f"Retrieved context:\n{self._retrieve(prompt, context)}\n\n{prompt}"},
        ]

        # Try the requested model first, then fall back — a model the account
        # cannot reach 404s, which previously surfaced as a silent empty stream.
        candidates = [model] if model else []
        candidates += [m for m in ([self.model] + OPENROUTER_MODELS) if m not in candidates]

        last_error = None
        for candidate in candidates:
            try:
                response = requests.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://woodlandltd.com"),
                        "X-Title": "Woodland OCR",
                    },
                    json={
                        "model": candidate,
                        "stream": True,
                        "temperature": 0.1,
                        "messages": messages,
                    },
                    timeout=90,
                    stream=True,
                )
            except requests.RequestException as error:
                last_error = f"{candidate}: {error}"
                continue

            if response.status_code != 200:
                # Read the body so the real reason reaches the logs instead of
                # being discarded as an empty stream.
                last_error = f"{candidate}: HTTP {response.status_code} {response.text[:200]}"
                response.close()
                continue

            yielded = False
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    choice = json.loads(payload).get("choices", [{}])[0]
                except json.JSONDecodeError:
                    continue
                delta = choice.get("delta", {}).get("content")
                if delta:
                    yielded = True
                    yield delta
                # A truncated answer is worth surfacing rather than ending quietly.
                if choice.get("finish_reason") == "length":
                    yield "\n\n[truncated: response hit the token limit]"

            if yielded:
                return
            last_error = f"{candidate}: stream produced no content"

        raise RuntimeError(f"OpenRouter streaming failed. Last error — {last_error}")


agent = WoodlandAgent()
