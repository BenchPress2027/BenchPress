"""Model providers.

A model is named as PROVIDER:MODEL_ID, e.g. openai:gpt-4o-mini,
anthropic:claude-sonnet-4-5, openrouter:mistralai/mistral-small-3.2-24b-instruct,
ollama:llama3.1:8b. Everything except Anthropic goes through the OpenAI Python
client against the provider's OpenAI-compatible endpoint.

Every call is a fresh single-turn conversation with the stimulus as the only
user message: no system prompt, no history. Temperature is not sent unless you
ask for one (the paper used each provider's default).
"""
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Provider:
    env_key: Optional[str]          # environment variable holding the API key (None: no key needed)
    base_url: Optional[str]         # OpenAI-compatible endpoint (None: the SDK default)
    sdk: str = "openai"             # "openai" or "anthropic"


PROVIDERS: Dict[str, Provider] = {
    "openai":     Provider("OPENAI_API_KEY", None),
    "anthropic":  Provider("ANTHROPIC_API_KEY", None, sdk="anthropic"),
    "google":     Provider("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "mistral":    Provider("MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
    "xai":        Provider("XAI_API_KEY", "https://api.x.ai/v1"),
    "groq":       Provider("GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    "together":   Provider("TOGETHER_API_KEY", "https://api.together.xyz/v1"),
    "openrouter": Provider("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "deepseek":   Provider("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    "ollama":     Provider(None, "http://localhost:11434/v1"),
    # any other OpenAI-compatible server (vLLM, LM Studio, TGI, an institutional gateway, ...):
    # pass --base-url, and --api-key-env if it needs a key
    "custom":     Provider(None, None),
}

# Models that think before answering: they need a large output budget, because the
# hidden reasoning counts against it. Matched on the model id.
_REASONING_RE = re.compile(r"(^|/)(o\d|gpt-5|gpt-oss)|deepseek-r1|deepseek-reasoner|qwq|-r1\b|thinking|reasoning",
                           re.IGNORECASE)
REASONING_MIN_MAX_TOKENS = 4096


def is_reasoning_model(model_id: str) -> bool:
    return bool(_REASONING_RE.search(model_id or ""))


def parse_model(spec: str):
    """'openai:gpt-4o' -> ('openai', 'gpt-4o'). Only the first ':' splits (ollama ids contain ':')."""
    if ":" not in spec:
        raise ValueError(f"Model {spec!r} must be given as PROVIDER:MODEL_ID, e.g. openai:gpt-4o-mini. "
                         f"Providers: {', '.join(PROVIDERS)}")
    provider, model_id = spec.split(":", 1)
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Known: {', '.join(PROVIDERS)}")
    if not model_id.strip():
        raise ValueError(f"Missing model id in {spec!r}")
    return provider, model_id.strip()


_PARAM_ERROR_HINTS = ("max_tokens", "max_completion_tokens", "temperature", "unsupported", "not supported")


class Client:
    """Sends one prompt, returns a dict with the response text and call metadata."""

    def __init__(self, spec: str, base_url: Optional[str] = None, api_key_env: Optional[str] = None,
                 timeout: float = 300.0):
        self.provider, self.model_id = parse_model(spec)
        p = PROVIDERS[self.provider]
        self.base_url = base_url or p.base_url
        if self.provider == "custom" and not self.base_url:
            raise ValueError("The 'custom' provider needs --base-url (e.g. http://localhost:8000/v1)")
        env_key = api_key_env or p.env_key
        api_key = os.environ.get(env_key) if env_key else None
        if env_key and not api_key:
            raise ValueError(f"Missing API key: set {env_key} in your environment or in a .env file "
                             f"(see .env.example).")
        if p.sdk == "anthropic":
            try:
                import anthropic
            except ImportError as e:
                raise ImportError("Anthropic models need the anthropic package: pip install anthropic") from e
            self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
            self._call = self._call_anthropic
        else:
            from openai import OpenAI
            # local servers accept any key, but the client refuses an empty one
            self._client = OpenAI(api_key=api_key or "not-needed", base_url=self.base_url,
                                  timeout=timeout, max_retries=0)
            self._call = self._call_openai
        self._token_param = "max_tokens"     # switched to max_completion_tokens on first rejection

    def call(self, prompt: str, max_tokens: int, temperature: Optional[float] = None) -> Dict[str, Any]:
        t0 = time.time()
        out = self._call(prompt, max_tokens, temperature)
        out["latency_ms"] = int(1000 * (time.time() - t0))
        return out

    def _call_openai(self, prompt, max_tokens, temperature):
        last = None
        for param in (self._token_param, "max_completion_tokens" if self._token_param == "max_tokens" else "max_tokens"):
            kwargs = {"model": self.model_id, "messages": [{"role": "user", "content": prompt}], param: max_tokens}
            if temperature is not None:
                kwargs["temperature"] = temperature
            try:
                r = self._client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "status_code", None)
                if status == 400 and any(h in str(e).lower() for h in _PARAM_ERROR_HINTS):
                    last = e
                    continue
                raise
            self._token_param = param
            choice = r.choices[0]
            msg = choice.message
            extra = getattr(msg, "model_extra", None) or {}
            reasoning = getattr(msg, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning")
            usage = r.usage
            return {"response": msg.content or "", "reasoning": reasoning,
                    "model_version": getattr(r, "model", None) or self.model_id,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "output_tokens": getattr(usage, "completion_tokens", None) if usage else None}
        raise last

    def _call_anthropic(self, prompt, max_tokens, temperature):
        kwargs = {"model": self.model_id, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if temperature is not None:
            kwargs["temperature"] = temperature
        r = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        return {"response": text, "reasoning": None, "model_version": getattr(r, "model", self.model_id),
                "finish_reason": getattr(r, "stop_reason", None),
                "input_tokens": getattr(r.usage, "input_tokens", None),
                "output_tokens": getattr(r.usage, "output_tokens", None)}


def is_retryable(exc: Exception) -> bool:
    """Rate limits, timeouts, connection problems and server errors are retried; bad requests are not."""
    status = getattr(exc, "status_code", None)
    if status is None:
        return True
    return status in (408, 409, 425, 429) or status >= 500
