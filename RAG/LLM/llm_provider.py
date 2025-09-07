# RAG/LLM/llm_provider.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Protocol

# -------- Interface (abstraction) --------
class LLMClient(Protocol):
    def ask_with_docs(self, question: str, docs: List[Any]) -> Dict[str, Any]:
        """Return {'answer': str, 'sources': docs}"""
        ...

# -------- DeepSeek via OpenRouter --------
class DeepSeekClient:
    def __init__(self, model_slug: str | None = None):
        from openai import OpenAI  # lazy import to avoid hard dep when not used
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set for DeepSeek/OpenRouter.")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model_slug or os.getenv("OR_MODEL_SLUG", "deepseek/deepseek-chat-v3-0324:free")

    def ask_with_docs(self, question: str, docs: List[Any]) -> Dict[str, Any]:
        context = "\n\n".join(
            f"Document {i + 1}:\n{getattr(d, 'page_content', str(d))}"
            for i, d in enumerate(docs)
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=700,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise, concise RAG assistant. "
                        "Use ONLY the provided context. If unknown, say you don't know. "
                        "Cite as [Doc i] when helpful."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
                },
            ],
        )
        return {"answer": completion.choices[0].message.content, "sources": docs}

# -------- Claude via Anthropic --------
class ClaudeClient:
    def __init__(self, model: str | None = None):
        import anthropic  # lazy import
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set for Claude/Anthropic.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

    def ask_with_docs(self, question: str, docs: List[Any]) -> Dict[str, Any]:
        context = "\n\n".join(
            f"Document {i + 1}: {getattr(d, 'page_content', str(d))}"
            for i, d in enumerate(docs)
        )
        prompt = f"""Answer this question based on the provided context:

Context:
{context}

Question: {question}

Answer:"""
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        return {"answer": msg.content[0].text, "sources": docs}

# -------- Factory (chooses implementation at startup) --------
def build_llm_from_env() -> LLMClient:
    provider = (os.getenv("LLM_PROVIDER") or "deepseek").lower()
    if provider == "deepseek":
        return DeepSeekClient()
    if provider == "claude":
        return ClaudeClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'deepseek' or 'claude'.")
