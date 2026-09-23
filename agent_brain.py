"""Woodland OCR agent brain: OpenRouter reasoning, local RAG, and MCP-style tools."""
import json
import os
from typing import Any, Dict, Optional

import requests


class WoodlandAgent:
    def __init__(self) -> None:
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

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
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
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

    def stream(self, prompt: str, context: Optional[Dict[str, Any]] = None, model: Optional[str] = None):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model or self.model,
                "stream": True,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": "You are Woodland OCR, a precise property-management assistant. Use only supplied context and never invent financial values."},
                    {"role": "user", "content": f"Retrieved context:\n{self._retrieve(prompt, context)}\n\n{prompt}"},
                ],
            },
            timeout=90,
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: ") and line[6:] != "[DONE]":
                try:
                    delta = json.loads(line[6:]).get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
                except json.JSONDecodeError:
                    continue


agent = WoodlandAgent()
