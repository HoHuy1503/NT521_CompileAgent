import logging
import os
import random
import re
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_FILE_PATH = os.path.join(REPO_ROOT, ".env")
load_dotenv(ENV_FILE_PATH)

LOCAL_LLM_PROVIDERS = ("ollama", "vllm", "lmstudio", "openai_compatible")


class FatalLLMError(RuntimeError):
    """Unrecoverable LLM/API error (all keys exhausted, invalid auth, or server offline)."""


def parse_api_keys(raw_value: str) -> List[str]:
    """Split a comma-separated API key string into deduplicated keys preserving order."""
    if not raw_value:
        return []
    keys: List[str] = []
    seen = set()
    for part in str(raw_value).split(","):
        cleaned = part.strip().strip('"').strip("'").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            keys.append(cleaned)
    return keys


def mask_api_key(value: str) -> str:
    """Return a masked representation of an API key suitable for logs."""
    if not value:
        return "<EMPTY>"
    if len(value) <= 12:
        return "***"
    return f"{value[:8]}...{value[-4:]}"


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
        # Never mistake HTTP errors raised by web-scraping `requests` tools for LLM API errors
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


_AUTH_TEXT_PATTERNS = (
    re.compile(r"\bHTTP(?:/\d(?:\.\d)?)?\s+(?:401|403)\b", re.IGNORECASE),
    re.compile(r"\bError\s+code\s*[:=]\s*(?:401|403)\b", re.IGNORECASE),
    re.compile(r"\bstatus[_\s]?code\s*[=:]\s*(?:401|403)\b", re.IGNORECASE),
    re.compile(r"\b401\s+Unauthorized\b", re.IGNORECASE),
)

_AUTH_SUBSTRINGS = (
    "api_key_invalid",
    "invalid_api_key",
    "incorrect api key",
    "invalid authentication",
    "missing api key for llm_provider",
    "authenticationerror",
    "permissiondeniederror",
)


def _is_auth_error(exc: BaseException) -> bool:
    """Return True only for genuine LLM API authentication/permission errors (no bare 401/403 substring checks)."""
    if _is_requests_web_error(exc):
        return False

    status_code = _extract_status_code(exc)
    if status_code in (401, 403):
        return True

    for cur in _iter_exception_chain(exc):
        mod = getattr(type(cur), "__module__", "") or ""
        if mod.startswith("requests"):
            continue
        if type(cur).__name__ in ("AuthenticationError", "PermissionDeniedError"):
            return True

    err_text = str(exc or "")
    err_low = err_text.lower()
    if any(tok in err_low for tok in _AUTH_SUBSTRINGS):
        return True
    if any(pat.search(err_text) for pat in _AUTH_TEXT_PATTERNS):
        return True
    return False


def _is_quota_exhausted_error(exc: BaseException) -> bool:
    """Return True when the active API key has exhausted its daily/billing quota."""
    if _is_requests_web_error(exc):
        return False
    err_text = str(exc or "")
    err_low = err_text.lower()
    explicit_daily_markers = (
        "perday",
        "per_day",
        "generaterequestsperday",
        "generatecontentinputtokenspermodelperday",
        "billing_hard_limit",
    )
    per_minute_markers = (
        "perminute",
        "per_minute",
        "requests per minute",
        "tokens per minute",
        "retry in ",
        "retrydelay",
        "rate_limit_exceeded",
    )
    has_per_minute = any(m in err_low for m in per_minute_markers)
    if any(m in err_low for m in explicit_daily_markers):
        return True
    if "check your plan and billing" in err_low and not has_per_minute:
        return True
    if ("insufficient_quota" in err_low or "exceeded your current quota" in err_low) and not has_per_minute:
        return True
    return False


def _is_extra_body_400_error(exc: BaseException) -> bool:
    """Return True when HTTP 400 indicates the endpoint rejects extra_body/chat_template_kwargs/options."""
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
        "unrecognized request argument",
        "extra inputs are not permitted",
    )
    return any(m in err_low for m in param_markers)


_RATE_LIMIT_PATTERNS = (
    re.compile(r"\bHTTP(?:/\d(?:\.\d)?)?\s+429\b", re.IGNORECASE),
    re.compile(r"\bError\s+(?:code\s*[:=]\s*)?429\b", re.IGNORECASE),
    re.compile(r"\bstatus[_\s]?code\s*[=:]\s*429\b", re.IGNORECASE),
    re.compile(r"\b429\s+Too\s+Many\s+Requests\b", re.IGNORECASE),
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True for transient RPM/TPM 429 rate-limit errors."""
    if _is_requests_web_error(exc) or _is_quota_exhausted_error(exc):
        return False
    if _extract_status_code(exc) == 429:
        return True
    for cur in _iter_exception_chain(exc):
        if type(cur).__name__ == "RateLimitError":
            return True
    err_text = str(exc or "")
    err_low = err_text.lower()
    if (
        "resource_exhausted" in err_low
        or "rate limit" in err_low
        or "too many requests" in err_low
    ):
        return True
    if any(pat.search(err_text) for pat in _RATE_LIMIT_PATTERNS):
        return True
    return False


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
    """Return True if exception indicates TCP/HTTP connection failure to the LLM endpoint."""
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
    """Return True for transient upstream 5xx / InternalServerError failures."""
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
    "is not found for api version",
    "model does not exist",
    "does not exist",
    "try pulling it first",
)


def _is_model_not_found_error(exc: BaseException) -> bool:
    """Return True for HTTP 404 / NotFoundError / model_not_found errors from LLM providers."""
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
    """Return True if exception is a transient LLM timeout, connection, 5xx, or 429 rate-limit error."""
    return (
        _is_timeout_error(exc)
        or _is_connection_error(exc)
        or _is_server_5xx_error(exc)
        or _is_rate_limit_error(exc)
    )


class KeyPoolManager:
    """Thread-safe API key pool supporting rotation on exhaustion, auth errors, and 429 limits."""

    def __init__(self):
        self._lock = threading.RLock()
        self.keys: List[str] = []
        self.active_index: int = 0
        self.exhausted_keys: set = set()
        self.cooldown_until: Dict[str, float] = {}
        self._quota_strikes: Dict[int, int] = {}
        self.signature: Optional[Tuple] = None
        self._clients: List[Tuple[int, weakref.ref]] = []

    def _initial_active_index(self) -> int:
        if len(self.keys) > 1 and os.environ.get("AC_MULTI_WORKER") == "1":
            return os.getpid() % len(self.keys)
        return 0

    def configure(self, keys: List[str], signature: Optional[Tuple] = None) -> None:
        with self._lock:
            self.keys = list(keys or [])
            self.active_index = self._initial_active_index()
            self.exhausted_keys = set()
            self.cooldown_until = {}
            self._quota_strikes = {}
            self.signature = signature
            if self.keys:
                self._sync_registered_clients(self.keys[self.active_index])

    def reset(self) -> None:
        with self._lock:
            self.active_index = self._initial_active_index()
            self.exhausted_keys = set()
            self.cooldown_until = {}
            self._quota_strikes = {}
            if self.keys:
                self._sync_registered_clients(self.keys[self.active_index])

    def clear_strikes(self) -> None:
        with self._lock:
            self._quota_strikes.clear()

    def strip_extra_body_on_clients(self) -> None:
        """Disable extra_body on registered LLM clients after an HTTP 400 unsupported-param error."""
        with self._lock:
            alive: List[Tuple[int, weakref.ref]] = []
            for cid, ref in self._clients:
                client = ref()
                if client is not None:
                    alive.append((cid, ref))
                    if hasattr(client, "extra_body"):
                        try:
                            setattr(client, "extra_body", None)
                        except Exception:
                            pass
                    mk = getattr(client, "model_kwargs", None)
                    if isinstance(mk, dict) and "extra_body" in mk:
                        try:
                            mk.pop("extra_body", None)
                        except Exception:
                            pass
            self._clients = alive

    def register_client(self, client) -> None:
        if client is None:
            return
        with self._lock:
            obj_id = id(client)
            alive: List[Tuple[int, weakref.ref]] = []
            already_registered = False
            for cid, ref in self._clients:
                if ref() is not None:
                    alive.append((cid, ref))
                    if cid == obj_id:
                        already_registered = True
            if not already_registered:
                try:
                    alive.append((obj_id, weakref.ref(client)))
                except TypeError:
                    pass
            self._clients = alive

    def _sync_registered_clients(self, new_key: str) -> None:
        _update_module_active_key(new_key)
        alive: List[Tuple[int, weakref.ref]] = []
        for cid, ref in self._clients:
            client = ref()
            if client is not None:
                alive.append((cid, ref))
                _apply_key_to_client(client, new_key)
        self._clients = alive

    def has_available_key(self) -> bool:
        with self._lock:
            if not self.keys:
                return False
            return any(k not in self.exhausted_keys for k in self.keys)

    def remaining_count(self) -> int:
        with self._lock:
            return sum(1 for k in self.keys if k not in self.exhausted_keys)

    def peek_active_key(self) -> str:
        with self._lock:
            if not self.keys:
                return ""
            if 0 <= self.active_index < len(self.keys):
                current = self.keys[self.active_index]
                if current not in self.exhausted_keys:
                    return current
            for idx, key in enumerate(self.keys):
                if key not in self.exhausted_keys:
                    self.active_index = idx
                    return key
            return ""

    def get_active_key(self) -> str:
        with self._lock:
            if not self.keys:
                return ""
            key = self.peek_active_key()
            if key:
                return key
            raise FatalLLMError(
                f"All {len(self.keys)} API key(s) in the pool are exhausted or invalid."
            )

    def mark_exhausted_and_rotate(
        self,
        failed_key: Optional[str] = None,
        reason: str = "",
        exc: Optional[BaseException] = None,
    ) -> Optional[str]:
        with self._lock:
            if not self.keys:
                return None
            if failed_key and failed_key in self.keys:
                old_idx = self.keys.index(failed_key)
                target = failed_key
            elif 0 <= self.active_index < len(self.keys):
                old_idx = self.active_index
                target = self.keys[old_idx]
            else:
                old_idx = 0
                target = self.keys[0]

            combined_low = f"{reason} {exc or ''}".lower()
            explicit_daily_markers = (
                "perday",
                "per_day",
                "generaterequestsperday",
                "generatecontentinputtokenspermodelperday",
                "billing_hard_limit",
            )
            has_explicit_daily = any(m in combined_low for m in explicit_daily_markers)
            is_429 = (
                (_extract_status_code(exc) == 429 if exc is not None else False)
                or ("429" in combined_low)
            )
            if not has_explicit_daily and is_429 and reason != "auth_error":
                strikes = self._quota_strikes.get(old_idx, 0) + 1
                self._quota_strikes[old_idx] = strikes
                if strikes < 2:
                    self.cooldown_until[target] = time.time() + 15.0
                    n = len(self.keys)
                    now = time.time()
                    chosen_idx = None
                    fallback_idx = None
                    for offset in range(1, n):
                        idx = (old_idx + offset) % n
                        candidate = self.keys[idx]
                        if candidate in self.exhausted_keys:
                            continue
                        if self.cooldown_until.get(candidate, 0.0) <= now:
                            chosen_idx = idx
                            break
                        if fallback_idx is None:
                            fallback_idx = idx
                    if chosen_idx is None:
                        chosen_idx = fallback_idx
                    if chosen_idx is not None:
                        self.active_index = chosen_idx
                        new_key = self.keys[chosen_idx]
                        logging.warning(
                            "[KeyPool] Ambiguous 429 on key %s (strike 1/2, %s); temporarily cooling down and rotating to key #%d/%d (%s).",
                            mask_api_key(target),
                            reason or "quota_suspect",
                            chosen_idx + 1,
                            n,
                            mask_api_key(new_key),
                        )
                        self._sync_registered_clients(new_key)
                        return new_key
                    logging.warning(
                        "[KeyPool] Ambiguous 429 on single key %s (strike 1/2); putting on temporary cooldown without permanent blacklist.",
                        mask_api_key(target),
                    )
                    return target

            self.exhausted_keys.add(target)

            n = len(self.keys)
            now = time.time()
            fallback_idx = None
            chosen_idx = None
            for offset in range(1, n + 1):
                idx = (self.active_index + offset) % n
                candidate = self.keys[idx]
                if candidate in self.exhausted_keys:
                    continue
                if self.cooldown_until.get(candidate, 0.0) <= now:
                    chosen_idx = idx
                    break
                if fallback_idx is None:
                    fallback_idx = idx

            if chosen_idx is None:
                chosen_idx = fallback_idx

            if chosen_idx is None:
                logging.error(
                    "[KeyPool] Exhausted key %s (%s). No remaining keys in pool.",
                    mask_api_key(target),
                    reason or "quota/auth",
                )
                return None

            self.active_index = chosen_idx
            new_key = self.keys[chosen_idx]
            logging.warning(
                "[KeyPool] Key %s unavailable (%s); rotating to key #%d/%d (%s).",
                mask_api_key(target),
                reason or "quota/auth",
                chosen_idx + 1,
                n,
                mask_api_key(new_key),
            )
            self._sync_registered_clients(new_key)
            return new_key

    def rotate_on_rate_limit(self, cooldown_seconds: float = 15.0) -> Tuple[str, bool]:
        """Rotate to another non-cooldown key on 429 if available; returns (key, rotated_fresh)."""
        with self._lock:
            if not self.keys:
                return "", False
            current = self.peek_active_key()
            if not current:
                raise FatalLLMError(
                    f"All {len(self.keys)} API key(s) in the pool are exhausted or invalid."
                )
            now = time.time()
            self.cooldown_until[current] = now + max(1.0, float(cooldown_seconds))

            n = len(self.keys)
            for offset in range(1, n):
                idx = (self.active_index + offset) % n
                candidate = self.keys[idx]
                if candidate in self.exhausted_keys:
                    continue
                if self.cooldown_until.get(candidate, 0.0) <= now:
                    self.active_index = idx
                    logging.info(
                        "[KeyPool] Rate-limited on %s; switching immediately to ready key #%d/%d (%s).",
                        mask_api_key(current),
                        idx + 1,
                        n,
                        mask_api_key(candidate),
                    )
                    self._sync_registered_clients(candidate)
                    return candidate, True

            # All available keys are currently cooling down: stay or pick earliest expiry, sleep required
            best_idx = self.active_index
            earliest = self.cooldown_until.get(current, now)
            for offset in range(1, n):
                idx = (self.active_index + offset) % n
                candidate = self.keys[idx]
                if candidate not in self.exhausted_keys:
                    cd = self.cooldown_until.get(candidate, 0.0)
                    if cd < earliest:
                        earliest = cd
                        best_idx = idx
            if best_idx != self.active_index:
                self.active_index = best_idx
                chosen = self.keys[best_idx]
                self._sync_registered_clients(chosen)
                return chosen, False
            return current, False


_KEY_POOL = KeyPoolManager()


def _apply_key_to_client(client, new_key: str) -> None:
    """Dynamically update API key credentials on an existing LangChain/OpenAI client."""
    try:
        from pydantic import SecretStr
    except ImportError:
        SecretStr = None

    try:
        if hasattr(client, "openai_api_key"):
            val = SecretStr(new_key) if SecretStr is not None else new_key
            setattr(client, "openai_api_key", val)
    except Exception:
        pass
    for attr in ("root_client", "root_async_client"):
        sub = getattr(client, attr, None)
        if sub is not None and hasattr(sub, "api_key"):
            try:
                sub.api_key = new_key
            except Exception:
                pass
    if hasattr(client, "api_key"):
        try:
            setattr(client, "api_key", new_key)
        except Exception:
            pass


def _update_module_active_key(new_key: str) -> None:
    """Keep module-level compatibility aliases synced with active key without mutating os.environ."""
    globals()["LLM_API_KEY"] = new_key
    globals()["OPENAI_API_KEY"] = new_key
    globals()["ANTHROPIC_API_KEY"] = new_key
    globals()["DEEPSEEK_API_KEY"] = new_key
    globals()["READMEAI_API_KEY"] = new_key


def get_key_pool() -> KeyPoolManager:
    """Return the active KeyPoolManager singleton (ensuring env config has been resolved)."""
    resolve_llm_config()
    return _KEY_POOL


def reset_key_pool() -> None:
    """Reset the API key pool rotation/exhaustion state from current environment."""
    cfg = resolve_llm_config()
    sig = (cfg["provider"], cfg["base_url"], tuple(cfg["api_keys"]))
    _KEY_POOL.configure(cfg["api_keys"], signature=sig)


def get_active_api_key() -> str:
    """Return the current active API key from the Key Pool, raising FatalLLMError if all exhausted."""
    resolve_llm_config()
    if not _KEY_POOL.keys:
        return ""
    return _KEY_POOL.get_active_key()


class _CircuitBreakerState:
    """Tracks consecutive LLM connectivity/timeout failures at both call and repo levels."""

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
    """Record a successful LLM call, resetting consecutive call failure and quota-strike counters."""
    _CIRCUIT_BREAKER.record_call_success()
    _KEY_POOL.clear_strikes()


def record_repo_llm_success() -> None:
    """Record that a repository finished without fatal/transient LLM endpoint failure."""
    _CIRCUIT_BREAKER.record_repo_success()


def probe_llm_server_health(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
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
    resolved_key = api_key if api_key is not None else (_KEY_POOL.peek_active_key() or cfg["api_key"])
    url = f"{resolved_base}/models"
    headers = {"Accept": "application/json"}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

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
            # An HTTP response (even 401/404) proves the server is alive, though 5xx means unhealthy
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
    Record an exhausted LLM transient/connectivity failure and evaluate the Circuit Breaker.
    Returns True if the batch should abort (FatalLLMError condition triggered).
    """
    cfg = resolve_llm_config()
    threshold = max(1, int(cfg.get("circuit_breaker_threshold", 3)))
    health_ok: Optional[bool] = None
    if check_health:
        health_ok = probe_llm_server_health(
            base_url=cfg["base_url"],
            api_key=_KEY_POOL.peek_active_key() or cfg["api_key"],
        )

    with _CIRCUIT_BREAKER._lock:
        _CIRCUIT_BREAKER.consecutive_call_failures += 1
        _CIRCUIT_BREAKER.last_error = str(exc or "")
        _CIRCUIT_BREAKER.last_health_ok = health_ok
        call_streak = _CIRCUIT_BREAKER.consecutive_call_failures
        repo_streak = _CIRCUIT_BREAKER.consecutive_repo_failures

    if health_ok is False:
        if cfg["provider"] in LOCAL_LLM_PROVIDERS:
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
        logging.critical(
            "[CircuitBreaker] LLM server health probe failed (%s); tripping circuit breaker.",
            cfg["base_url"],
        )
        return True
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


def is_fatal_llm_error(exc: BaseException) -> bool:
    """Return True for unrecoverable LLM errors that must stop the batch rather than record fake failures."""
    if isinstance(exc, FatalLLMError):
        return True
    if _is_model_not_found_error(exc):
        return True
    if _is_quota_exhausted_error(exc) or _is_auth_error(exc):
        resolve_llm_config()
        # Fatal only if no backup keys remain in the pool (or pool is empty)
        return _KEY_POOL.remaining_count() <= 1
    # Single connection/timeout errors are NOT immediately fatal on their own;
    # Circuit Breaker / health probe promotes them to FatalLLMError when systemic.
    return False


def _compute_rate_limit_sleep(err_text: str, attempt: int) -> float:
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_text, re.IGNORECASE)
    if not m:
        m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", err_text)
    if m:
        return min(max(float(m.group(1)) + random.uniform(1.0, 3.0), 3.0), 70.0)
    base = min(8.0 * (2 ** min(attempt, 4)), 60.0)
    return base + random.uniform(0.5, 3.0)


def call_with_rate_limit_retry(fn, *args, max_attempts: Optional[int] = None, **kwargs):
    """Execute fn with multi-key rotation, backoff+jitter retry, and Circuit Breaker health protection."""
    cfg = resolve_llm_config()
    attempts_limit = (
        int(max_attempts)
        if max_attempts is not None
        else max(1, int(os.getenv("LLM_RETRY_ATTEMPTS", "4") or "4"))
    )
    transient_attempt = 0
    key_switches = 0
    max_key_switches = max(len(_KEY_POOL.keys), 1) * 2
    extra_body_stripped = False

    while True:
        try:
            result = fn(*args, **kwargs)
            record_llm_success()
            return result
        except Exception as exc:
            if isinstance(exc, FatalLLMError):
                raise
            # Avoid double-retrying or double-counting Circuit Breaker if an outer caller wraps an inner retry
            if getattr(exc, "_llm_retry_exhausted", False):
                raise

            err_text = str(exc or "")
            err_name = type(exc).__name__

            if not extra_body_stripped and _is_extra_body_400_error(exc):
                extra_body_stripped = True
                logging.warning(
                    "[!] Endpoint rejected extra_body/thinking params with HTTP 400 (%s); "
                    "stripping extra_body and retrying once.",
                    err_text[:180],
                )
                _KEY_POOL.strip_extra_body_on_clients()
                bound_self = getattr(fn, "__self__", None)
                if bound_self is not None:
                    if hasattr(bound_self, "extra_body"):
                        try:
                            setattr(bound_self, "extra_body", None)
                        except Exception:
                            pass
                    mk = getattr(bound_self, "model_kwargs", None)
                    if isinstance(mk, dict) and "extra_body" in mk:
                        try:
                            mk.pop("extra_body", None)
                        except Exception:
                            pass
                if "extra_body" in kwargs:
                    kwargs.pop("extra_body", None)
                continue

            if _is_model_not_found_error(exc):
                logging.critical("[!] Fatal model configuration error (%s): %s", err_name, err_text)
                raise FatalLLMError(err_text) from exc

            if _is_quota_exhausted_error(exc) or _is_auth_error(exc):
                reason = "quota_exhausted" if _is_quota_exhausted_error(exc) else "auth_error"
                if key_switches < max_key_switches and _KEY_POOL.keys:
                    prev_key = _KEY_POOL.peek_active_key()
                    new_key = _KEY_POOL.mark_exhausted_and_rotate(reason=reason, exc=exc)
                    if new_key is not None:
                        key_switches += 1
                        if new_key == prev_key and len(_KEY_POOL.keys) == 1:
                            time.sleep(min(_compute_rate_limit_sleep(err_text, key_switches - 1), 5.0))
                        continue
                raise FatalLLMError(err_text) from exc

            if _is_rate_limit_error(exc):
                wait_seconds = _compute_rate_limit_sleep(err_text, transient_attempt)
                if len(_KEY_POOL.keys) > 1 and key_switches < max_key_switches:
                    _, rotated_fresh = _KEY_POOL.rotate_on_rate_limit(cooldown_seconds=wait_seconds)
                    if rotated_fresh:
                        key_switches += 1
                        continue

                if transient_attempt >= attempts_limit - 1:
                    try:
                        setattr(exc, "_llm_retry_exhausted", True)
                    except Exception:
                        pass
                    if record_llm_failure(exc, check_health=True):
                        raise FatalLLMError(err_text) from exc
                    raise
                logging.warning(
                    "[!] Hit API RPM/TPM rate limit (attempt %d/%d); sleeping %.1fs...",
                    transient_attempt + 1,
                    attempts_limit,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                transient_attempt += 1
                continue

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
                    "[!] Transient LLM error (%s: %s) (attempt %d/%d); sleeping %.1fs...",
                    err_name,
                    err_text[:120],
                    transient_attempt + 1,
                    attempts_limit,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                transient_attempt += 1
                continue

            # Unrecognized/programming/parsing errors propagate unchanged
            raise


# LangSmith config
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
LOG_URL_TEMPLATE = os.getenv("LOG_URL_TEMPLATE", "")

# Unified OpenAI-compatible LLM provider presets and configuration
_PROVIDER_PRESETS = {
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "master_model": "gemini-2.5-flash",
        "discussion_models": None,
        "embedding_model": "gemini-embedding-001",
        "timeout": 60,
        "api_key_fallback": "",
        "requires_api_key": True,
        "context_window": None,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "master_model": "qwen3:30b",
        "discussion_models": None,
        "embedding_model": "nomic-embed-text",
        "timeout": 180,
        "api_key_fallback": "ollama",
        "requires_api_key": False,
        "context_window": 16384,
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "master_model": "Qwen/Qwen3-30B-A3B",
        "discussion_models": None,
        "embedding_model": "BAAI/bge-m3",
        "timeout": 180,
        "api_key_fallback": "EMPTY",
        "requires_api_key": False,
        "context_window": None,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "master_model": "qwen/qwen3-30b-a3b",
        "discussion_models": None,
        "embedding_model": "text-embedding-nomic-embed-text-v1.5",
        "timeout": 180,
        "api_key_fallback": "lm-studio",
        "requires_api_key": False,
        "context_window": None,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "master_model": "gpt-4o-mini",
        "discussion_models": ["gpt-4o-mini"],
        "embedding_model": "text-embedding-3-small",
        "timeout": 60,
        "api_key_fallback": "",
        "requires_api_key": True,
        "context_window": None,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "master_model": "qwen/qwen3-30b-a3b",
        "discussion_models": None,
        "embedding_model": "openai/text-embedding-3-small",
        "timeout": 90,
        "api_key_fallback": "",
        "requires_api_key": True,
        "context_window": None,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "master_model": "deepseek-chat",
        "discussion_models": ["deepseek-chat"],
        "embedding_model": "",
        "timeout": 90,
        "api_key_fallback": "",
        "requires_api_key": True,
        "context_window": None,
    },
    "openai_compatible": {
        "base_url": "http://localhost:8000/v1",
        "master_model": "qwen3-30b",
        "discussion_models": None,
        "embedding_model": "",
        "timeout": 180,
        "api_key_fallback": "EMPTY",
        "requires_api_key": False,
        "context_window": None,
    },
}

_PROVIDER_ALIASES = {
    "gemini": "google",
    "local": "ollama",
    "custom": "openai_compatible",
    "openai-compatible": "openai_compatible",
}


def resolve_thinking_mode(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    raw_thinking: Optional[str] = None,
) -> str:
    """Resolve effective thinking mode ('auto' | 'true' | 'false') for a target model/provider."""
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

    # 'auto': disable verbose <think> on local Qwen3/DeepSeek-R1/QwQ ReAct workloads (ollama/vllm/lmstudio)
    prov = (provider or "").strip().lower()
    mdl = (model or "").strip().lower()
    if prov in ("ollama", "vllm", "lmstudio") and any(
        tok in mdl for tok in ("qwen3", "deepseek-r1", "qwq")
    ):
        return "false"
    return "auto"


def resolve_llm_config(provider: str = None) -> dict:
    """Resolve active LLM endpoint, key pool credentials, models, timeouts, and local tuning from env."""
    raw_provider = (provider or os.getenv("LLM_PROVIDER") or "google").strip().lower()
    canonical = _PROVIDER_ALIASES.get(raw_provider, raw_provider)
    if canonical not in _PROVIDER_PRESETS:
        supported = ", ".join(sorted(_PROVIDER_PRESETS.keys()))
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{raw_provider}'. Supported providers: {supported}"
        )

    preset = _PROVIDER_PRESETS[canonical]
    explicit_base_url = os.getenv("LLM_BASE_URL", "").strip()
    if canonical == "google":
        base_url = explicit_base_url or os.getenv("GEMINI_BASE_URL", "").strip() or preset["base_url"]
    elif canonical == "openai":
        base_url = explicit_base_url or os.getenv("OPENAI_BASE_URL", "").strip() or preset["base_url"]
    elif canonical == "deepseek":
        base_url = explicit_base_url or os.getenv("DEEPSEEK_BASE_URL", "").strip() or preset["base_url"]
    elif canonical == "openai_compatible":
        base_url = explicit_base_url or os.getenv("OPENAI_BASE_URL", "").strip() or preset["base_url"]
    else:
        base_url = explicit_base_url or preset["base_url"]

    normalized_base_url = base_url.rstrip("/")

    key_candidates = [os.getenv("LLM_API_KEYS", ""), os.getenv("LLM_API_KEY", "")]
    if canonical == "google":
        key_candidates.append(os.getenv("GOOGLE_API_KEY", ""))
    elif canonical == "openai":
        key_candidates.append(os.getenv("OPENAI_API_KEY", ""))
    elif canonical == "deepseek":
        key_candidates.append(os.getenv("DEEPSEEK_API_KEY", ""))
    elif canonical == "openrouter":
        key_candidates.extend([os.getenv("OPENROUTER_API_KEY", ""), os.getenv("OPENAI_API_KEY", "")])
    elif canonical in ("vllm", "openai_compatible"):
        key_candidates.append(os.getenv("OPENAI_API_KEY", ""))

    api_keys: List[str] = []
    for raw_k in key_candidates:
        parsed = parse_api_keys(raw_k)
        if parsed:
            api_keys = parsed
            break
    if not api_keys and preset["api_key_fallback"]:
        api_keys = [preset["api_key_fallback"]]

    sig = (canonical, normalized_base_url, tuple(api_keys))
    if _KEY_POOL.signature != sig:
        _KEY_POOL.configure(api_keys, signature=sig)

    active_key = _KEY_POOL.peek_active_key()

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

    solver_mode = (os.getenv("ERROR_SOLVER_MODE", "auto") or "auto").strip().lower()
    if solver_mode not in ("auto", "fast", "debate"):
        solver_mode = "auto"

    cb_env = os.getenv("LLM_CIRCUIT_BREAKER_THRESHOLD", "").strip()
    circuit_breaker_threshold = max(1, int(cb_env)) if cb_env else 3

    nav_iter_env = os.getenv("NAVIGATOR_MAX_ITERATIONS", "").strip()
    navigator_max_iterations = max(1, int(nav_iter_env)) if nav_iter_env else 10

    return {
        "provider": canonical,
        "base_url": normalized_base_url,
        "api_key": active_key,
        "api_keys": list(api_keys),
        "master_model": master_model,
        "discussion_models": discussion_models,
        "embedding_model": embedding_model,
        "timeout": timeout,
        "requires_api_key": preset["requires_api_key"],
        "context_window": context_window,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "error_solver_mode": solver_mode,
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
        and "qwen3" in target_model
        and "/no_think" not in text
    ):
        return f"{text.rstrip()}\n/no_think"
    return text


_REACT_SALVAGE_MARKER_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:\*{0,2}Final\s+Answer\*{0,2}\s*:|\*{0,2}Action\s*\d*\*{0,2}\s*:)",
    re.IGNORECASE,
)


def _salvage_react_from_text(tail: str) -> str:
    """Salvage a complete Action/Action Input or Final Answer block from an unclosed <think> section."""
    if not tail:
        return ""
    # Check for Final Answer first (from last occurrence onward)
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

    # Check for Action + Action Input pair (from last Action match onward)
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


def create_chat_llm(
    model: str = None,
    temperature: float = None,
    timeout: int = None,
    max_retries: int = 0,
    base_url: str = None,
    api_key: str = None,
    extra_body: dict = None,
    max_tokens: int = None,
):
    """Instantiate a ChatOpenAI client bound to dynamic KeyPool/provider config."""
    from langchain_openai import ChatOpenAI

    cfg = resolve_llm_config()
    resolved_base_url = (base_url or cfg["base_url"]).rstrip("/")

    use_pool = api_key is None
    if api_key is not None:
        resolved_api_key = api_key
    else:
        if not cfg["api_keys"] and cfg["requires_api_key"]:
            raise ValueError(
                f"Missing API key for LLM_PROVIDER='{cfg['provider']}'. "
                f"Set LLM_API_KEYS / LLM_API_KEY (or GOOGLE_API_KEY for google) in {ENV_FILE_PATH}."
            )
        resolved_api_key = get_active_api_key() if cfg["api_keys"] else ""

    if not resolved_api_key and cfg["requires_api_key"]:
        raise ValueError(
            f"Missing API key for LLM_PROVIDER='{cfg['provider']}'. "
            f"Set LLM_API_KEYS / LLM_API_KEY (or GOOGLE_API_KEY for google) in {ENV_FILE_PATH}."
        )

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
    is_thinking_model = any(tok in resolved_model.lower() for tok in ("qwen3", "deepseek-r1"))
    if (
        thinking_mode == "false"
        and is_thinking_model
        and (
            cfg["provider"] in ("ollama", "vllm", "lmstudio")
            or (cfg["provider"] == "openai_compatible" and explicit_disable)
        )
    ):
        ctk = dict(merged_extra_body.get("chat_template_kwargs") or {})
        ctk.setdefault("enable_thinking", False)
        merged_extra_body["chat_template_kwargs"] = ctk

    llm_kwargs = {
        "base_url": resolved_base_url,
        "model": resolved_model,
        "api_key": resolved_api_key or "EMPTY",
        "temperature": resolved_temp,
        "timeout": resolved_timeout,
        "max_retries": max_retries,
    }
    if resolved_max_tokens is not None:
        llm_kwargs["max_tokens"] = int(resolved_max_tokens)
    if merged_extra_body:
        llm_kwargs["extra_body"] = merged_extra_body

    llm = ChatOpenAI(**llm_kwargs)
    if use_pool:
        _KEY_POOL.register_client(llm)
    return llm


_ACTIVE_LLM_CONFIG = resolve_llm_config()
LLM_PROVIDER = _ACTIVE_LLM_CONFIG["provider"]
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
LLM_BASE_URL = _ACTIVE_LLM_CONFIG["base_url"]
LLM_API_KEYS = _ACTIVE_LLM_CONFIG["api_keys"]
LLM_API_KEY = _ACTIVE_LLM_CONFIG["api_key"]
GEMINI_BASE_URL = LLM_BASE_URL
MASTER_MODEL = _ACTIVE_LLM_CONFIG["master_model"]
DISCUSSION_MODELS = _ACTIVE_LLM_CONFIG["discussion_models"]

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT = _ACTIVE_LLM_CONFIG["timeout"]
LLM_CONTEXT_WINDOW = _ACTIVE_LLM_CONFIG["context_window"]
LLM_MAX_TOKENS = _ACTIVE_LLM_CONFIG["max_tokens"]
LLM_THINKING = _ACTIVE_LLM_CONFIG["thinking"]
ERROR_SOLVER_MODE = _ACTIVE_LLM_CONFIG["error_solver_mode"]
LLM_CIRCUIT_BREAKER_THRESHOLD = _ACTIVE_LLM_CONFIG["circuit_breaker_threshold"]
NAVIGATOR_MAX_ITERATIONS = _ACTIVE_LLM_CONFIG["navigator_max_iterations"]

USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
MAKE_JOBS = int(os.getenv("MAKE_JOBS", "4"))

# Backward-compatible aliases for downstream modules
OPENAI_BASE_URL = LLM_BASE_URL
OPENAI_MODEL = DISCUSSION_MODELS[0]
OPENAI_EMBEDDING_MODEL = _ACTIVE_LLM_CONFIG["embedding_model"]
OPENAI_API_KEY = LLM_API_KEY

ANTHROPIC_BASE_URL = LLM_BASE_URL
ANTHROPIC_MODEL = DISCUSSION_MODELS[1] if len(DISCUSSION_MODELS) > 1 else DISCUSSION_MODELS[0]
ANTHROPIC_API_KEY = LLM_API_KEY

DEEPSEEK_BASE_URL = LLM_BASE_URL
DEEPSEEK_MODEL = MASTER_MODEL
DEEPSEEK_API_KEY = LLM_API_KEY

READMEAI_BASE_URL = LLM_BASE_URL
READMEAI_MODEL = MASTER_MODEL
READMEAI_API_KEY = LLM_API_KEY

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


Begin!
Question: {input}

Thought:{agent_scratchpad}"""


model_decision_template = """I want you to help me compile the project {project_name} from source code. Please complete the compilation task step by step.

The project has been downloaded into /work/, which is also the Shell's initial path.
You should follow these rules:
1. Use the Shell tool to execute any command for compilation, and it is recommended to use `tree . -L 2` command to show the project structure.
2. Use the CompileNavigator tool to looking for compilation instructions. Try to avoid reading documents in Shell tool, unless you think it's necessary.
3. If a build or configuration command fails with missing headers/libraries, dependency errors, or compiler/linker errors that you cannot immediately resolve, call the ErrorSolver tool with the error details to obtain a verified fix.
4. APT package lists are not pre-cached in this container: whenever installing packages or missing libraries, always run `apt-get update && apt-get install -y <package>`.
5. No need to install this project or test it, just compile it.
6. Compilation target should be the main body of the project, for example, the executable programs for a tool-type project or the shared/static libraries for a library project.
7. When you've done your best, you need to check if the project actually compiled successfully and output COMPILATION-SUCCESS, COMPILATION-FAIL, or COMPILATION-UNCERTAIN as the final answer.
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
