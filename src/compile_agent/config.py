import logging
import os
import random
import re
import threading
import time
from typing import Optional, Tuple
from dotenv import load_dotenv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_FILE_PATH = os.path.join(REPO_ROOT, ".env")
load_dotenv(ENV_FILE_PATH)

LOCAL_LLM_PROVIDERS = ("llamacpp", "ollama", "vllm", "lmstudio", "openai_compatible")


class FatalLLMError(RuntimeError):
    """Unrecoverable local LLM endpoint error (model missing or server persistently offline)."""


def _iter_exception_chain(exc: BaseException):
    """Yield exc and shallow linked causes/contexts without cycling."""
    visited = set()
    cur = exc
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        yield cur
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)


def _extract_status_code(exc: BaseException) -> Optional[int]:
    """Extract an HTTP status code from non-web-scraper exceptions."""
    for cur in _iter_exception_chain(exc):
        mod = getattr(type(cur), "__module__", "") or ""
        if mod.startswith("requests"):
            continue
        code = getattr(cur, "status_code", None)
        if code is None:
            resp = getattr(cur, "response", None)
            code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _is_requests_web_error(exc: BaseException) -> bool:
    """Return True if exception originates from the `requests` HTTP client (e.g. web scraping)."""
    for cur in _iter_exception_chain(exc):
        mod = getattr(type(cur), "__module__", "") or ""
        if mod.startswith("requests"):
            return True
    return False


def _is_extra_body_400_error(exc: BaseException) -> bool:
    """Return True when HTTP 400 indicates the local endpoint rejects extra_body/chat_template_kwargs/options."""
    if _is_requests_web_error(exc):
        return False
    status_code = _extract_status_code(exc)
    err_text = str(exc or "")
    err_low = err_text.lower()
    err_name_low = type(exc).__name__.lower()
    is_400 = (status_code == 400) or ("400" in err_text) or ("badrequest" in err_name_low)
    if not is_400:
        return False
    param_markers = (
        "extra_body",
        "chat_template_kwargs",
        "options",
        "enable_thinking",
        "reasoning_format",
        "reasoning_budget",
        "unrecognized request argument",
        "extra inputs are not permitted",
        "unknown field",
        "invalid parameter",
    )
    return any(m in err_low for m in param_markers)


def _is_timeout_error(exc: BaseException) -> bool:
    """Return True if exception represents a request/read/connect timeout."""
    if _is_requests_web_error(exc):
        return False
    for cur in _iter_exception_chain(exc):
        if type(cur).__name__ in ("APITimeoutError", "ConnectTimeout", "ReadTimeout", "TimeoutError"):
            return True
    err_low = str(exc or "").lower()
    return "request timed out" in err_low or "timed out" in err_low


def _is_connection_error(exc: BaseException) -> bool:
    """Return True if exception indicates TCP/HTTP connection failure to the local LLM endpoint."""
    if _is_requests_web_error(exc):
        return False
    for cur in _iter_exception_chain(exc):
        if type(cur).__name__ in ("APIConnectionError", "ConnectError", "ConnectionError"):
            return True
    err_low = str(exc or "").lower()
    return (
        "connection error" in err_low
        or "connection refused" in err_low
        or "failed to establish a new connection" in err_low
    )


def _is_server_5xx_error(exc: BaseException) -> bool:
    """Return True for transient local server 5xx / InternalServerError failures (e.g. model loading)."""
    if _is_requests_web_error(exc):
        return False
    status_code = _extract_status_code(exc)
    if status_code in (500, 502, 503, 504):
        return True
    for cur in _iter_exception_chain(exc):
        if type(cur).__name__ in ("InternalServerError", "BadGatewayError", "ServiceUnavailableError"):
            return True
    err_text = str(exc or "")
    return bool(
        re.search(
            r"\b(?:HTTP(?:/\d(?:\.\d)?)?\s+50[0234]|Error\s+code\s*[:=]\s*50[0234]|502\s+Bad\s+Gateway|503\s+Service\s+Unavailable)\b",
            err_text,
            re.IGNORECASE,
        )
    )


_MODEL_NOT_FOUND_TEXT_PATTERNS = (
    re.compile(r"\bHTTP(?:/\d(?:\.\d)?)?\s+404\b", re.IGNORECASE),
    re.compile(r"\bError\s+(?:code\s*[:=]\s*)?404\b", re.IGNORECASE),
    re.compile(r"\bstatus[_\s]?code\s*[=:]\s*404\b", re.IGNORECASE),
    re.compile(r"\bmodel\s+[`'\"]?[^\s`'\"]+[`'\"]?\s+(?:not\s+found|does\s+not\s+exist)\b", re.IGNORECASE),
)

_MODEL_NOT_FOUND_SUBSTRINGS = (
    "model_not_found",
    "model not found",
    "model does not exist",
    "does not exist",
    "try pulling it first",
)


def _is_model_not_found_error(exc: BaseException) -> bool:
    """Return True for HTTP 404 / NotFoundError / model_not_found errors from local LLM servers."""
    if _is_requests_web_error(exc):
        return False
    if _extract_status_code(exc) == 404:
        return True
    for cur in _iter_exception_chain(exc):
        mod = getattr(type(cur), "__module__", "") or ""
        if mod.startswith("requests"):
            continue
        if type(cur).__name__ == "NotFoundError":
            return True
    err_text = str(exc or "")
    err_low = err_text.lower()
    if any(tok in err_low for tok in _MODEL_NOT_FOUND_SUBSTRINGS):
        return True
    if any(pat.search(err_text) for pat in _MODEL_NOT_FOUND_TEXT_PATTERNS):
        return True
    return False


def is_transient_llm_error(exc: BaseException) -> bool:
    """Return True if exception is a transient local LLM timeout, connection, or 5xx error."""
    return (
        _is_timeout_error(exc)
        or _is_connection_error(exc)
        or _is_server_5xx_error(exc)
    )


def is_fatal_llm_error(exc: BaseException) -> bool:
    """Return True for unrecoverable LLM errors (missing model or tripped Circuit Breaker)."""
    if isinstance(exc, FatalLLMError):
        return True
    return _is_model_not_found_error(exc)


class _CircuitBreakerState:
    """Tracks consecutive local LLM connectivity/timeout failures at both call and repo levels."""

    def __init__(self):
        self._lock = threading.RLock()
        self.consecutive_call_failures: int = 0
        self.consecutive_repo_failures: int = 0
        self.last_error: str = ""
        self.last_health_ok: Optional[bool] = None

    def reset(self) -> None:
        with self._lock:
            self.consecutive_call_failures = 0
            self.consecutive_repo_failures = 0
            self.last_error = ""
            self.last_health_ok = None

    def record_call_success(self) -> None:
        with self._lock:
            self.consecutive_call_failures = 0
            self.last_error = ""
            self.last_health_ok = True

    def record_repo_success(self) -> None:
        with self._lock:
            self.consecutive_call_failures = 0
            self.consecutive_repo_failures = 0
            self.last_error = ""
            self.last_health_ok = True

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "consecutive_call_failures": self.consecutive_call_failures,
                "consecutive_repo_failures": self.consecutive_repo_failures,
                "last_error": self.last_error,
                "last_health_ok": self.last_health_ok,
            }


_CIRCUIT_BREAKER = _CircuitBreakerState()


def reset_circuit_breaker() -> None:
    """Reset Circuit Breaker failure counters."""
    _CIRCUIT_BREAKER.reset()


def get_circuit_breaker_state() -> dict:
    """Return a copy of the Circuit Breaker counters."""
    return _CIRCUIT_BREAKER.snapshot()


def record_llm_success() -> None:
    """Record a successful local LLM call, resetting consecutive call failure counter."""
    _CIRCUIT_BREAKER.record_call_success()


def record_repo_llm_success() -> None:
    """Record that a repository finished without fatal/transient LLM endpoint failure."""
    _CIRCUIT_BREAKER.record_repo_success()


def probe_llm_server_health(
    base_url: Optional[str] = None,
    timeout: float = 10.0,
) -> bool:
    """Perform a lightweight GET {base_url}/models health check (returns True on HTTP 2xx)."""
    import urllib.error
    import urllib.parse
    import urllib.request

    cfg = resolve_llm_config()
    resolved_base = (base_url or cfg["base_url"] or "").rstrip("/")
    if not resolved_base:
        return False
    url = f"{resolved_base}/models"
    headers = {"Accept": "application/json"}

    parsed = urllib.parse.urlparse(resolved_base)
    host = (parsed.hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    else:
        use_proxy_env = os.getenv("USE_PROXY", "").strip().lower()
        use_proxy_global = bool(globals().get("USE_PROXY", False))
        use_proxy = (use_proxy_env == "true") if use_proxy_env else use_proxy_global
        proxy_url = ""
        if use_proxy:
            proxy_url = (
                os.getenv("PROXY", "").strip()
                or (globals().get("PROXY") or "").strip()
            )
        if proxy_url:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
            )
        else:
            opener = urllib.request.build_opener()

    req = urllib.request.Request(url, headers=headers, method="GET")
    for attempt in range(2):
        try:
            with opener.open(req, timeout=float(timeout)) as resp:
                status = getattr(resp, "status", 200)
                return 200 <= int(status) < 300
        except urllib.error.HTTPError as exc:
            if exc.code in (500, 502, 503, 504):
                logging.warning("[HealthProbe] %s returned HTTP %s", url, exc.code)
                return False
            return exc.code < 500
        except Exception as exc:
            if attempt == 0:
                time.sleep(2.0)
                continue
            logging.warning("[HealthProbe] %s unreachable after retry: %s", url, exc)
            return False
    return False


def record_llm_failure(exc: BaseException, check_health: bool = True) -> bool:
    """
    Record an exhausted local LLM transient/connectivity failure and evaluate the Circuit Breaker.
    Returns True if the batch should abort (FatalLLMError condition triggered).
    """
    cfg = resolve_llm_config()
    threshold = max(1, int(cfg.get("circuit_breaker_threshold", 3)))
    health_ok: Optional[bool] = None
    if check_health:
        health_ok = probe_llm_server_health(base_url=cfg["base_url"])

    with _CIRCUIT_BREAKER._lock:
        _CIRCUIT_BREAKER.consecutive_call_failures += 1
        _CIRCUIT_BREAKER.last_error = str(exc or "")
        _CIRCUIT_BREAKER.last_health_ok = health_ok
        call_streak = _CIRCUIT_BREAKER.consecutive_call_failures
        repo_streak = _CIRCUIT_BREAKER.consecutive_repo_failures

    if health_ok is False:
        if call_streak >= 2 or repo_streak >= 1:
            logging.critical(
                "[CircuitBreaker] Local LLM server health probe failed (%s) with call_streak=%d, repo_streak=%d; tripping circuit breaker.",
                cfg["base_url"],
                call_streak,
                repo_streak,
            )
            return True
        logging.warning(
            "[CircuitBreaker] Local LLM health probe to %s timed out on first failure (call_streak=%d, repo_streak=%d); waiting for confirmation before aborting batch.",
            cfg["base_url"],
            call_streak,
            repo_streak,
        )
        return False

    if health_ok is True:
        if repo_streak >= threshold and call_streak >= threshold * 2:
            logging.critical(
                "[CircuitBreaker] Consecutive LLM failures reached systemic threshold with healthy probe "
                "(call_streak=%d/%d, repo_streak=%d/%d).",
                call_streak,
                threshold * 2,
                repo_streak,
                threshold,
            )
            return True
    elif call_streak >= threshold or repo_streak >= threshold:
        logging.critical(
            "[CircuitBreaker] Consecutive LLM failures reached threshold (%d/%d, repo_streak=%d).",
            call_streak,
            threshold,
            repo_streak,
        )
        return True

    logging.warning(
        "[CircuitBreaker] Transient failure (call_streak=%d, repo_streak=%d, threshold=%d, health_ok=%s; %s) — continuing batch.",
        call_streak,
        repo_streak,
        threshold,
        health_ok,
        cfg["base_url"],
    )
    return False


def record_repo_llm_failure(exc: BaseException) -> bool:
    """
    Increment repo-level LLM failure streak when a repo aborts due to LLM connectivity/timeout.
    Raises FatalLLMError if threshold is reached or returns False otherwise.
    """
    if isinstance(exc, FatalLLMError):
        raise exc
    if not is_transient_llm_error(exc):
        return False

    cfg = resolve_llm_config()
    threshold = max(1, int(cfg.get("circuit_breaker_threshold", 3)))
    with _CIRCUIT_BREAKER._lock:
        _CIRCUIT_BREAKER.consecutive_repo_failures += 1
        repo_streak = _CIRCUIT_BREAKER.consecutive_repo_failures
        _CIRCUIT_BREAKER.last_error = str(exc or "")

    if repo_streak >= threshold:
        raise FatalLLMError(
            f"Circuit breaker tripped after {repo_streak} consecutive repository LLM failures: {exc}"
        ) from exc
    return False


_EXTRA_BODY_LOCK = threading.RLock()
_EXTRA_BODY_UNSUPPORTED_ENDPOINTS: set = set()


def _strip_extra_body_on_obj(obj) -> None:
    if obj is None:
        return
    if hasattr(obj, "extra_body"):
        try:
            setattr(obj, "extra_body", None)
        except Exception:
            pass
    mk = getattr(obj, "model_kwargs", None)
    if isinstance(mk, dict) and "extra_body" in mk:
        try:
            mk.pop("extra_body", None)
        except Exception:
            pass


def call_with_llm_retry(fn, *args, max_attempts: Optional[int] = None, **kwargs):
    """Execute fn with transient local-server backoff+jitter retry and Circuit Breaker protection."""
    resolve_llm_config()
    attempts_limit = (
        int(max_attempts)
        if max_attempts is not None
        else max(1, int(os.getenv("LLM_RETRY_ATTEMPTS", "4") or "4"))
    )
    transient_attempt = 0
    extra_body_stripped = False

    while True:
        try:
            result = fn(*args, **kwargs)
            record_llm_success()
            return result
        except Exception as exc:
            if isinstance(exc, FatalLLMError):
                raise
            if getattr(exc, "_llm_retry_exhausted", False):
                raise

            err_text = str(exc or "")
            err_name = type(exc).__name__

            if not extra_body_stripped and _is_extra_body_400_error(exc):
                extra_body_stripped = True
                logging.warning(
                    "[!] Local endpoint rejected extra_body/thinking params with HTTP 400 (%s); "
                    "stripping extra_body and retrying once.",
                    err_text[:180],
                )
                bound_self = getattr(fn, "__self__", None)
                _strip_extra_body_on_obj(bound_self)
                if bound_self is not None:
                    base = str(getattr(bound_self, "openai_api_base", "") or "").rstrip("/")
                    mdl = str(getattr(bound_self, "model_name", "") or "")
                    if base or mdl:
                        with _EXTRA_BODY_LOCK:
                            _EXTRA_BODY_UNSUPPORTED_ENDPOINTS.add((base, mdl))
                if "extra_body" in kwargs:
                    kwargs.pop("extra_body", None)
                continue

            if _is_model_not_found_error(exc):
                logging.critical("[!] Fatal local model configuration error (%s): %s", err_name, err_text)
                raise FatalLLMError(err_text) from exc

            if is_transient_llm_error(exc):
                if transient_attempt >= attempts_limit - 1:
                    try:
                        setattr(exc, "_llm_retry_exhausted", True)
                    except Exception:
                        pass
                    if record_llm_failure(exc, check_health=True):
                        raise FatalLLMError(err_text) from exc
                    raise
                wait_seconds = min(3.0 * (2 ** min(transient_attempt, 4)), 30.0) + random.uniform(
                    0.2, 1.5
                )
                logging.warning(
                    "[!] Transient local LLM error (%s: %s) (attempt %d/%d); sleeping %.1fs...",
                    err_name,
                    err_text[:120],
                    transient_attempt + 1,
                    attempts_limit,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                transient_attempt += 1
                continue

            raise


# Backward-compatible alias for internal callers
call_with_rate_limit_retry = call_with_llm_retry

LOG_URL_TEMPLATE = os.getenv("LOG_URL_TEMPLATE", "")

# Local OpenAI-compatible LLM backend presets (no API keys required)
_PROVIDER_PRESETS = {
    "llamacpp": {
        "base_url": "http://localhost:8080/v1",
        "master_model": "Qwen 3.8 K_M Q4 27B",
        "discussion_models": None,
        "embedding_model": "",
        "timeout": 240,
        "context_window": 16384,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "master_model": "qwen3:30b",
        "discussion_models": None,
        "embedding_model": "nomic-embed-text",
        "timeout": 180,
        "context_window": 16384,
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "master_model": "Qwen/Qwen3-30B-A3B",
        "discussion_models": None,
        "embedding_model": "BAAI/bge-m3",
        "timeout": 180,
        "context_window": None,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "master_model": "qwen/qwen3-30b-a3b",
        "discussion_models": None,
        "embedding_model": "text-embedding-nomic-embed-text-v1.5",
        "timeout": 180,
        "context_window": None,
    },
    "openai_compatible": {
        "base_url": "http://localhost:8000/v1",
        "master_model": "qwen3-27b",
        "discussion_models": None,
        "embedding_model": "",
        "timeout": 180,
        "context_window": None,
    },
}

_PROVIDER_ALIASES = {
    "local": "ollama",
    "llama.cpp": "llamacpp",
    "llama_cpp": "llamacpp",
    "llama-cpp": "llamacpp",
    "llama-server": "llamacpp",
    "custom": "openai_compatible",
    "openai-compatible": "openai_compatible",
}

_QWEN3_MODEL_RE = re.compile(r"qwen[\s._-]*3", re.IGNORECASE)
_THINKING_MODEL_TOKENS = (
    "deepseek-r1",
    "deepseek_r1",
    "qwq",
    "thinking",
    "reasoning",
    "gpt-oss",
    "glm-4.5",
    "glm-z1",
)


def _is_qwen3_model_name(model: Optional[str]) -> bool:
    """Return True if model identifier refers to a Qwen 3.x family model."""
    return bool(_QWEN3_MODEL_RE.search(str(model or "")))


def _is_thinking_model_name(model: Optional[str]) -> bool:
    """Return True if model identifier matches known reasoning/thinking model families."""
    mdl = str(model or "").strip().lower()
    if not mdl:
        return False
    if _is_qwen3_model_name(mdl):
        return True
    return any(tok in mdl for tok in _THINKING_MODEL_TOKENS)


def resolve_thinking_mode(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    raw_thinking: Optional[str] = None,
) -> str:
    """Resolve effective thinking mode ('auto' | 'true' | 'false') for a target local model/provider."""
    mode = (raw_thinking if raw_thinking is not None else os.getenv("LLM_THINKING", "auto") or "auto").strip().lower()
    if mode in ("true", "1", "yes", "on"):
        return "true"
    if mode in ("false", "0", "no", "off"):
        return "false"

    disable_env = os.getenv("LLM_DISABLE_THINKING", "").strip().lower()
    if disable_env in ("1", "true", "yes", "on"):
        return "false"
    if disable_env in ("0", "false", "no", "off"):
        return "true"

    prov = (provider or "").strip().lower()
    if prov in LOCAL_LLM_PROVIDERS and (_is_thinking_model_name(model) or prov == "llamacpp"):
        return "false"
    return "auto"


def resolve_react_stop_sequences():
    """Return stop_sequence setting for create_react_agent (uses colon-delimited '\\nObservation:' by default)."""
    raw = os.getenv("LLM_REACT_STOP", "").strip().lower()
    if raw in ("0", "false", "no", "off", "none"):
        return False
    return ["\nObservation:"]


def resolve_llm_config(provider: str = None) -> dict:
    """Resolve active local LLM endpoint, models, timeouts, and context tuning from env."""
    raw_provider = (provider or os.getenv("LLM_PROVIDER") or "llamacpp").strip().lower()
    canonical = _PROVIDER_ALIASES.get(raw_provider, raw_provider)
    if canonical not in _PROVIDER_PRESETS:
        supported = ", ".join(sorted(_PROVIDER_PRESETS.keys()))
        raise ValueError(
            f"Unsupported local LLM_PROVIDER='{raw_provider}'. Supported providers: {supported}"
        )

    preset = _PROVIDER_PRESETS[canonical]
    explicit_base_url = os.getenv("LLM_BASE_URL", "").strip()
    base_url = explicit_base_url or os.getenv("OPENAI_BASE_URL", "").strip() or preset["base_url"]
    normalized_base_url = base_url.rstrip("/")

    master_env = os.getenv("MASTER_MODEL", "").strip()
    master_model = master_env or preset["master_model"]
    discussion_env = os.getenv("DISCUSSION_MODELS", "").strip()
    if discussion_env:
        discussion_models = [m.strip() for m in discussion_env.split(",") if m.strip()]
    elif master_env:
        discussion_models = [master_model]
    elif preset["discussion_models"]:
        discussion_models = list(preset["discussion_models"])
    else:
        discussion_models = [master_model]

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip() or preset["embedding_model"]
    timeout_env = os.getenv("LLM_TIMEOUT", "").strip()
    timeout = int(timeout_env) if timeout_env else preset["timeout"]

    ctx_env = os.getenv("LLM_CONTEXT_WINDOW", "").strip()
    context_window = int(ctx_env) if ctx_env else preset.get("context_window")

    max_tok_env = os.getenv("LLM_MAX_TOKENS", "").strip()
    max_tokens = int(max_tok_env) if max_tok_env and int(max_tok_env) > 0 else None

    thinking = resolve_thinking_mode(model=master_model, provider=canonical)

    cb_env = os.getenv("LLM_CIRCUIT_BREAKER_THRESHOLD", "").strip()
    circuit_breaker_threshold = max(1, int(cb_env)) if cb_env else 3

    nav_iter_env = os.getenv("NAVIGATOR_MAX_ITERATIONS", "").strip()
    navigator_max_iterations = max(1, int(nav_iter_env)) if nav_iter_env else 15

    return {
        "provider": canonical,
        "base_url": normalized_base_url,
        "master_model": master_model,
        "discussion_models": discussion_models,
        "embedding_model": embedding_model,
        "timeout": timeout,
        "context_window": context_window,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "circuit_breaker_threshold": circuit_breaker_threshold,
        "navigator_max_iterations": navigator_max_iterations,
    }


def apply_thinking_directive(text: str, model: Optional[str] = None) -> str:
    """Append /no_think for Qwen3 local prompts when LLM_THINKING resolves to 'false'."""
    if not text:
        return text
    cfg = resolve_llm_config()
    target_model = (model or cfg["master_model"] or "").lower()
    effective = resolve_thinking_mode(model=target_model, provider=cfg["provider"])
    if (
        effective == "false"
        and cfg["provider"] in LOCAL_LLM_PROVIDERS
        and _is_qwen3_model_name(target_model)
        and "/no_think" not in text
    ):
        return f"{text.rstrip()}\n/no_think"
    return text


def _salvage_react_from_text(tail: str) -> str:
    """Salvage a complete Action/Action Input or Final Answer block from an unclosed <think> section."""
    if not tail:
        return ""
    fa_matches = list(
        re.finditer(r"(?:^|\n)[ \t]*\*{0,2}Final\s+Answer\*{0,2}\s*:\s*\S", tail, re.IGNORECASE)
    )
    if fa_matches:
        start = fa_matches[-1].start()
        if tail[start : start + 1] == "\n":
            start += 1
        candidate = tail[start:].strip()
        candidate = re.split(r"\n\s*Observation\s*:", candidate, maxsplit=1)[0].strip()
        return candidate

    action_matches = list(
        re.finditer(r"(?:^|\n)[ \t]*\*{0,2}Action\s*\d*\*{0,2}\s*:", tail, re.IGNORECASE)
    )
    for m in reversed(action_matches):
        start = m.start()
        if tail[start : start + 1] == "\n":
            start += 1
        segment = tail[start:].strip()
        if re.search(
            r"^\*{0,2}Action\s*\d*\*{0,2}\s*:\s*[^\n]+\n[\s\S]*?\*{0,2}Action\s*\d*\s*Input\s*\d*\*{0,2}\s*:\s*\S",
            segment,
            re.IGNORECASE,
        ):
            segment = re.split(r"\n\s*Observation\s*:", segment, maxsplit=1)[0].strip()
            return segment
    return ""


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks, salvaging valid ReAct Actions from unclosed/truncated blocks."""
    if not text:
        return ""
    raw = str(text)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)

    if "<think>" not in raw.lower() and "</think>" in cleaned.lower():
        if not re.match(
            r"^(?:\*{0,2}Thought\*{0,2}\s*:)?\s*(?:\*{0,2}Action\s*\d*\*{0,2}\s*:|\*{0,2}Final\s+Answer\*{0,2}\s*:)",
            cleaned.lstrip(),
            flags=re.IGNORECASE,
        ):
            cleaned = re.sub(
                r"^.*?</think>",
                "",
                cleaned,
                count=1,
                flags=re.DOTALL | re.IGNORECASE,
            )

    unclosed_match = re.search(r"<think>(.*)$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if unclosed_match:
        prefix = cleaned[: unclosed_match.start()].strip()
        tail = unclosed_match.group(1)
        salvaged = _salvage_react_from_text(tail)
        if salvaged:
            cleaned = f"{prefix}\n{salvaged}".strip() if prefix else salvaged
        else:
            cleaned = prefix

    return cleaned.strip()


_STOP_BYPASS_LOCK = threading.RLock()
_STOP_BYPASS_ENDPOINTS: set = set()
_REASONING_SAFE_CHAT_OPENAI_CLS = None


def reset_stop_bypass_cache() -> None:
    """Clear remembered (base_url, model) endpoints that drop content when stop cuts reasoning_content."""
    with _STOP_BYPASS_LOCK:
        _STOP_BYPASS_ENDPOINTS.clear()


def _extract_choice_reasoning(raw_choice: dict, raw_obj_choice=None) -> str:
    """Extract non-empty reasoning_content / reasoning text from an OpenAI-compatible choice."""
    if isinstance(raw_choice, dict):
        msg = raw_choice.get("message")
        if isinstance(msg, dict):
            for key in ("reasoning_content", "reasoning"):
                val = msg.get(key)
                if isinstance(val, str) and val.strip():
                    return val
    if raw_obj_choice is not None:
        obj_msg = getattr(raw_obj_choice, "message", None)
        if obj_msg is not None:
            for key in ("reasoning_content", "reasoning"):
                val = getattr(obj_msg, key, None)
                if isinstance(val, str) and val.strip():
                    return val
            extra = getattr(obj_msg, "model_extra", None)
            if isinstance(extra, dict):
                for key in ("reasoning_content", "reasoning"):
                    val = extra.get(key)
                    if isinstance(val, str) and val.strip():
                        return val
    return ""


def _get_reasoning_safe_chat_openai_cls():
    """Lazily build a ChatOpenAI subclass that handles llama.cpp/vLLM reasoning_content & stop truncation."""
    global _REASONING_SAFE_CHAT_OPENAI_CLS
    if _REASONING_SAFE_CHAT_OPENAI_CLS is not None:
        return _REASONING_SAFE_CHAT_OPENAI_CLS

    from langchain_openai import ChatOpenAI

    class ReasoningSafeChatOpenAI(ChatOpenAI):
        """ChatOpenAI subclass that salvages reasoning_content and recovers from stop-truncated empty turns."""

        def _endpoint_key(self) -> Tuple[str, str]:
            base = str(getattr(self, "openai_api_base", "") or "").rstrip("/")
            mdl = str(getattr(self, "model_name", "") or "")
            return (base, mdl)

        def _create_chat_result(self, response, generation_info=None):
            result = super()._create_chat_result(response, generation_info)
            try:
                response_dict = response if isinstance(response, dict) else response.model_dump()
            except Exception:
                response_dict = {}
            raw_choices = response_dict.get("choices") if isinstance(response_dict, dict) else None
            obj_choices = getattr(response, "choices", None) if not isinstance(response, dict) else None
            if not raw_choices or not result.generations:
                return result

            for idx, gen in enumerate(result.generations):
                if idx >= len(raw_choices):
                    break
                raw_obj = obj_choices[idx] if (obj_choices and idx < len(obj_choices)) else None
                reasoning = _extract_choice_reasoning(raw_choices[idx], raw_obj)
                if not reasoning:
                    continue
                gen.message.additional_kwargs["reasoning_content"] = reasoning
                content_str = str(gen.message.content or "")
                if not content_str.strip():
                    salvaged = _salvage_react_from_text(reasoning)
                    if salvaged:
                        gen.message.content = salvaged
                        gen.message.additional_kwargs["reasoning_salvaged"] = True
                    else:
                        gen.message.content = f"<think>{reasoning}"
                        gen.message.additional_kwargs["empty_content_with_reasoning"] = True
            return result

        def _should_retry_without_stop(self, result, effective_stop) -> Tuple[bool, bool]:
            """Return (should_retry, remember_endpoint_bypass) for empty stop-truncated responses."""
            if not effective_stop or not result.generations:
                return False, False
            gen0 = result.generations[0]
            msg0 = gen0.message
            gen_info = gen0.generation_info or {}
            finish_reason = str(gen_info.get("finish_reason") or "").lower()
            if finish_reason == "length":
                return False, False
            if msg0.additional_kwargs.get("empty_content_with_reasoning"):
                return True, True
            if not str(msg0.content or "").strip():
                return True, False
            return False, False

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            endpoint_key = self._endpoint_key()
            effective_stop = stop
            if effective_stop:
                with _STOP_BYPASS_LOCK:
                    if endpoint_key in _STOP_BYPASS_ENDPOINTS:
                        effective_stop = None

            with _EXTRA_BODY_LOCK:
                if endpoint_key in _EXTRA_BODY_UNSUPPORTED_ENDPOINTS:
                    _strip_extra_body_on_obj(self)

            try:
                result = super()._generate(
                    messages, stop=effective_stop, run_manager=run_manager, **kwargs
                )
            except Exception as exc:
                if _is_extra_body_400_error(exc) and (
                    getattr(self, "extra_body", None) is not None
                    or isinstance(getattr(self, "model_kwargs", None), dict)
                    and "extra_body" in getattr(self, "model_kwargs", {})
                ):
                    with _EXTRA_BODY_LOCK:
                        _EXTRA_BODY_UNSUPPORTED_ENDPOINTS.add(endpoint_key)
                    _strip_extra_body_on_obj(self)
                    result = super()._generate(
                        messages, stop=effective_stop, run_manager=run_manager, **kwargs
                    )
                else:
                    raise

            should_retry, remember_bypass = self._should_retry_without_stop(result, effective_stop)
            if should_retry:
                if remember_bypass:
                    with _STOP_BYPASS_LOCK:
                        _STOP_BYPASS_ENDPOINTS.add(endpoint_key)
                    logging.warning(
                        "[!] LLM endpoint %s (%s) returned empty content with reasoning_content under stop=%s; "
                        "retrying without stop sequence and bypassing stop for this endpoint.",
                        endpoint_key[0],
                        endpoint_key[1],
                        effective_stop,
                    )
                else:
                    logging.warning(
                        "[!] LLM endpoint %s (%s) returned empty content under stop=%s; retrying once without stop sequence.",
                        endpoint_key[0],
                        endpoint_key[1],
                        effective_stop,
                    )
                result = super()._generate(
                    messages, stop=None, run_manager=run_manager, **kwargs
                )
            return result

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            endpoint_key = self._endpoint_key()
            effective_stop = stop
            if effective_stop:
                with _STOP_BYPASS_LOCK:
                    if endpoint_key in _STOP_BYPASS_ENDPOINTS:
                        effective_stop = None

            with _EXTRA_BODY_LOCK:
                if endpoint_key in _EXTRA_BODY_UNSUPPORTED_ENDPOINTS:
                    _strip_extra_body_on_obj(self)

            result = await super()._agenerate(
                messages, stop=effective_stop, run_manager=run_manager, **kwargs
            )
            should_retry, remember_bypass = self._should_retry_without_stop(result, effective_stop)
            if should_retry:
                if remember_bypass:
                    with _STOP_BYPASS_LOCK:
                        _STOP_BYPASS_ENDPOINTS.add(endpoint_key)
                result = await super()._agenerate(
                    messages, stop=None, run_manager=run_manager, **kwargs
                )
            return result

    _REASONING_SAFE_CHAT_OPENAI_CLS = ReasoningSafeChatOpenAI
    return _REASONING_SAFE_CHAT_OPENAI_CLS


def create_chat_llm(
    model: str = None,
    temperature: float = None,
    timeout: int = None,
    max_retries: int = 0,
    base_url: str = None,
    extra_body: dict = None,
    max_tokens: int = None,
):
    """Instantiate a ReasoningSafeChatOpenAI client pointing to the active Local LLM endpoint."""
    ChatOpenAICls = _get_reasoning_safe_chat_openai_cls()

    cfg = resolve_llm_config()
    resolved_base_url = (base_url or cfg["base_url"]).rstrip("/")
    resolved_model = model or cfg["master_model"]
    resolved_temp = LLM_TEMPERATURE if temperature is None else temperature
    resolved_timeout = cfg["timeout"] if timeout is None else timeout
    resolved_max_tokens = cfg["max_tokens"] if max_tokens is None else max_tokens

    merged_extra_body = dict(extra_body) if extra_body else {}
    if cfg["provider"] == "ollama" and cfg.get("context_window"):
        opts = dict(merged_extra_body.get("options") or {})
        opts.setdefault("num_ctx", int(cfg["context_window"]))
        merged_extra_body["options"] = opts

    thinking_mode = resolve_thinking_mode(model=resolved_model, provider=cfg["provider"])
    explicit_disable = (
        os.getenv("LLM_DISABLE_THINKING", "").strip().lower() in ("1", "true", "yes", "on")
        or os.getenv("LLM_THINKING", "").strip().lower() in ("0", "false", "no", "off")
    )
    is_thinking_model = _is_thinking_model_name(resolved_model)
    if (
        thinking_mode == "false"
        and cfg["provider"] in LOCAL_LLM_PROVIDERS
        and (is_thinking_model or explicit_disable or cfg["provider"] == "llamacpp")
    ):
        ctk = dict(merged_extra_body.get("chat_template_kwargs") or {})
        ctk.setdefault("enable_thinking", False)
        merged_extra_body["chat_template_kwargs"] = ctk

    with _EXTRA_BODY_LOCK:
        if (resolved_base_url, resolved_model) in _EXTRA_BODY_UNSUPPORTED_ENDPOINTS:
            merged_extra_body.pop("chat_template_kwargs", None)
            merged_extra_body.pop("options", None)

    llm_kwargs = {
        "base_url": resolved_base_url,
        "model": resolved_model,
        "api_key": "EMPTY",
        "temperature": resolved_temp,
        "timeout": resolved_timeout,
        "max_retries": max_retries,
    }
    if resolved_max_tokens is not None:
        llm_kwargs["max_tokens"] = int(resolved_max_tokens)
    if merged_extra_body:
        llm_kwargs["extra_body"] = merged_extra_body

    return ChatOpenAICls(**llm_kwargs)


_ACTIVE_LLM_CONFIG = resolve_llm_config()
LLM_PROVIDER = _ACTIVE_LLM_CONFIG["provider"]
LLM_BASE_URL = _ACTIVE_LLM_CONFIG["base_url"]
MASTER_MODEL = _ACTIVE_LLM_CONFIG["master_model"]
DISCUSSION_MODELS = _ACTIVE_LLM_CONFIG["discussion_models"]

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT = _ACTIVE_LLM_CONFIG["timeout"]
LLM_CONTEXT_WINDOW = _ACTIVE_LLM_CONFIG["context_window"]
LLM_MAX_TOKENS = _ACTIVE_LLM_CONFIG["max_tokens"]
LLM_THINKING = _ACTIVE_LLM_CONFIG["thinking"]
LLM_CIRCUIT_BREAKER_THRESHOLD = _ACTIVE_LLM_CONFIG["circuit_breaker_threshold"]
NAVIGATOR_MAX_ITERATIONS = _ACTIVE_LLM_CONFIG["navigator_max_iterations"]

USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
MAKE_JOBS = int(os.getenv("MAKE_JOBS", "4"))

OPENAI_BASE_URL = LLM_BASE_URL
OPENAI_MODEL = DISCUSSION_MODELS[0]
OPENAI_EMBEDDING_MODEL = _ACTIVE_LLM_CONFIG["embedding_model"]

PROXY = os.getenv("PROXY", "") if USE_PROXY else ""
PASSWORD = os.getenv("PASSWORD", "root")

DEFAULT_HTTP_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def build_http_headers() -> dict:
    """Build HTTP request headers with safe fallback if UserAgent lookup fails."""
    user_agent = DEFAULT_HTTP_USER_AGENT
    try:
        from fake_useragent import UserAgent

        user_agent = UserAgent().random or DEFAULT_HTTP_USER_AGENT
    except Exception:
        pass
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def resolve_http_proxies(proxy_cfg=None):
    """Normalize boolean or string proxy settings into a requests-compatible dict."""
    if isinstance(proxy_cfg, str):
        proxy_url = proxy_cfg.strip()
    elif proxy_cfg:
        proxy_url = PROXY.strip() if isinstance(PROXY, str) else ""
    else:
        proxy_url = ""
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}


# Logs
PROCESS_LOG_CSV_PATH = os.path.join(REPO_ROOT, "outputs", "runs")
FLOW_BASED_LOG_CSV_PATH = ""

# To compile projects
COMPILE_PROJECTS_PATH = ""

# ========================== agent config ==========================
template = """You are an experienced software development engineer. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [ {tool_names} ]
Action Input: the input to the action, do not explain the input further
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important: Never write "Observation:" yourself. Stop immediately after "Action Input:".

Begin!
Question: {input}

Thought:{agent_scratchpad}"""


model_decision_template = """I want you to help me compile the project {project_name} from source code. Please complete the compilation task step by step.

The project has been downloaded into /work/, which is also the Shell's initial path.
You should follow these rules:
1. Use the Shell tool to execute any command for compilation, and it is recommended to use `tree . -L 2` command to show the project structure.
2. Use the CompileNavigator tool to looking for compilation instructions. Try to avoid reading documents in Shell tool, unless you think it's necessary.
3. If there are any difficulties during compilation, try to break them without calling tools.
4. Unless you encounter a problem that you cannot solve, there is no need to call the ErrorSolver tool.
5. No need to install this project or test it, just compile it.
6. Compilation target should be the main body of the project, for example, the executable programs for a tool-type project or the shared/static libraries for a library project.
7. When you've done your best, you need to check if the project actually compiled successfully and output COMPILATION-SUCCESS, COMPILATION-FAIL, or COMPILATION-UNCERTAIN as the final answer.
8. APT package lists are not pre-cached in this container: whenever installing packages or missing libraries, always run `apt-get update && apt-get install -y <package>`.
"""


def build_decision_question(
    project_name: str,
    disable_navigator: bool = False,
    disable_error_solver: bool = False,
) -> str:
    if not disable_navigator and not disable_error_solver:
        return apply_thinking_directive(model_decision_template.format(project_name=project_name))

    rules = [
        "1. Use the Shell tool to execute any command for compilation, and it is recommended to use `tree . -L 2` command to show the project structure."
    ]
    step = 2
    if not disable_navigator:
        rules.append(
            f"{step}. Use the CompileNavigator tool to looking for compilation instructions. Try to avoid reading documents in Shell tool, unless you think it's necessary."
        )
        step += 1
    else:
        rules.append(
            f"{step}. Inspect project files (such as README, CMakeLists.txt, Makefile, configure.ac) directly via the Shell tool to determine the build instructions."
        )
        step += 1

    if not disable_error_solver:
        rules.append(
            f"{step}. If a build or configuration command fails with missing headers/libraries, dependency errors, or compiler/linker errors that you cannot immediately resolve, call the ErrorSolver tool with the error details to obtain a verified fix."
        )
        step += 1
    else:
        rules.append(
            f"{step}. Analyze any compilation or configuration errors directly using the Shell tool output and fix missing dependencies or build flags on your own."
        )
        step += 1

    rules.extend(
        [
            f"{step}. APT package lists are not pre-cached in this container: whenever installing packages or missing libraries, always run `apt-get update && apt-get install -y <package>`.",
            f"{step + 1}. No need to install this project or test it, just compile it.",
            f"{step + 2}. Compilation target should be the main body of the project, for example, the executable programs for a tool-type project or the shared/static libraries for a library project.",
            f"{step + 3}. When you've done your best, you need to check if the project actually compiled successfully and output COMPILATION-SUCCESS, COMPILATION-FAIL, or COMPILATION-UNCERTAIN as the final answer.",
        ]
    )
    rules_block = "\n".join(rules)
    question_text = (
        f"I want you to help me compile the project {project_name} from source code. Please complete the compilation task step by step.\n\n"
        f"The project has been downloaded into /work/, which is also the Shell's initial path.\n"
        f"You should follow these rules:\n{rules_block}\n"
    )
    return apply_thinking_directive(question_text)


discussion_template1 = """You are an experienced compiler troubleshooting expert. Your task is to analyze provided error messages, identify likely causes, and deliver specific solutions. 

The entire compilation process was performed on Debian GNU/Linux 12 (bookworm) with GCC 13.2.0 and root user. Note that APT package indices are NOT cached in this container, so any apt installation command must include `apt-get update && apt-get install -y <pkg>`.
the error messages during {project} compilation are as follows:
{input}

Do NOT output ReAct Action/Action Input lines. Respond ONLY with the requested JSON object.
{tools}

Output the answer in JSON format as follows (Strict JSON, NO Additional Text):
{{
    "reasoning": "Concise explanation of problem-solving process",
    "solution": "Specific instructions, requiring conciseness, professionalism and accuracy(No additional relevant texts)",
    "confidence_level": <numeric confidence between 0.0 and 1.0>
}}

Begin!
"""

discussion_template2 = """You are an experienced compiler troubleshooting expert. Your task is to analyze provided error messages, identify likely causes, and deliver specific solutions.

The entire compilation process was performed on Debian GNU/Linux 12 (bookworm) with GCC 13.2.0 and root user. Note that APT package indices are NOT cached in this container, so any apt installation command must include `apt-get update && apt-get install -y <pkg>`.
the error messages during {project} compilation are as follows:
{input}

Do NOT output ReAct Action/Action Input lines. Respond ONLY with the requested JSON object.
{tools}

Carefully review the following solutions from other agents as additional information, and provide your own answer. Clearly states that which pointview do you agree or disagree and why.
{chat_history}

Output the answer in JSON format as follows (Strict JSON, NO Additional Text):
{{
    "reasoning": "Concise explanation of problem-solving process",
    "solution": "Specific instructions, requiring conciseness, professionalism and accuracy(No additional relevant texts)",
    "confidence_level": <numeric confidence between 0.0 and 1.0>
}}

Begin!
"""

get_instructions_template = """You are an experienced software development engineer. I want you to help me find out the project {project_name} compilation instructions. Please complete the searching task step by step, use the tools provided if require (one tool each time).

Project structure: {project_structure}

You should follow these rules:
1. Analyze the given project structure and reference the document file which exists compilation instructions, and then use DEBATE tool to check it.
2. Use the SEARCH_INSTRUCTIONS_FROM_FILES tool to looking for compilation instructions within the file inferred in the previous step. 
3. If URLs are found, fetch their content using GET_CONTENT_FROM_URL.
4. Compile all information into clear compilation instructions, do not include test commands.
5. Keep the instructions concise, professional and accurate.
"""
