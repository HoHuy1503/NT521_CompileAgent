"""Environment and Local LLM backend verification script."""

import importlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from compile_agent.config import (
    LOCAL_LLM_PROVIDERS,
    apply_thinking_directive,
    resolve_llm_config,
    strip_thinking_tags,
)

REQUIRED_PACKAGES = [
    "dotenv",
    "openai",
    "langchain",
    "langchain_openai",
    "paramiko",
    "docker",
    "bs4",
    "duckduckgo_search",
    "tenacity",
]

OPTIONAL_PACKAGES = [
    "chromadb",
]


def load_env_file(env_path: str) -> None:
    """Load variables from .env using python-dotenv or a minimal fallback."""
    if load_dotenv is not None:
        load_dotenv(env_path, override=True)
        return
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def run_cmd(cmd: list, timeout: int = 15) -> tuple:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"Command timed out after {timeout}s"


def check_python_and_packages() -> None:
    """Check Python version and required/optional Python dependencies."""
    print("=== 1. Python & Packages ===")
    print(f"Python: {sys.version.splitlines()[0]} ({sys.executable})")
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"  [INSTALLED] {pkg}")
        except ImportError:
            print(f"  [MISSING]   {pkg}")
    for pkg in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"  [INSTALLED] {pkg} (optional)")
        except ImportError:
            print(f"  [OPTIONAL]  {pkg} (not installed; not needed unless Chroma RAG is enabled)")


def check_docker() -> None:
    """Check Docker CLI, daemon status, and autocompiler:gcc13 image."""
    print("\n=== 2. Docker Desktop ===")
    rc, out, err = run_cmd(["docker", "--version"])
    print(f"docker --version (rc={rc}): {out or err}")

    rc, out, err = run_cmd(
        ["docker", "info", "--format", "{{.OSType}} | {{.ServerVersion}} | {{.OperatingSystem}}"]
    )
    print(f"docker info (rc={rc}): {out or err}")

    rc, out, err = run_cmd(["docker", "ps"])
    if rc == 0:
        lines = out.splitlines()
        print(f"docker ps (rc=0): {max(0, len(lines) - 1)} running container(s)")
    else:
        print(f"docker ps (rc={rc}): {err or out}")

    rc, out, err = run_cmd(["docker", "images", "-q", "autocompiler:gcc13"])
    image_status = out if (rc == 0 and out) else "<NOT BUILT>"
    print(f"image autocompiler:gcc13 (rc={rc}): {image_status}")


def http_json(url: str, headers: dict, payload: dict = None, timeout: int = 20) -> tuple:
    """Send a JSON GET/POST request (bypassing proxy on localhost) and return (status_code, body)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    method = "POST" if payload is not None else "GET"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    else:
        opener = urllib.request.build_opener()

    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, raw[:300]
    except Exception as exc:
        return 0, str(exc)


def check_llm_provider() -> None:
    """Test whichever Local LLM server/endpoint is active in .env (no API key required)."""
    print("\n=== 3. Active Local LLM Provider & Endpoint ===")
    try:
        cfg = resolve_llm_config()
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return

    print(f"LLM_PROVIDER:      {cfg['provider']}")
    print(f"LLM_BASE_URL:      {cfg['base_url']}")
    print(f"MASTER_MODEL:      {cfg['master_model']}")
    print(f"DISCUSSION_MODELS: {', '.join(cfg['discussion_models'])}")
    print(f"LLM_TIMEOUT:       {cfg['timeout']}s")
    print(
        f"TUNING:            ctx={cfg.get('context_window') or 'default'}, "
        f"max_tokens={cfg.get('max_tokens') or 'default'}, "
        f"thinking={cfg.get('thinking')}, "
        f"cb_threshold={cfg.get('circuit_breaker_threshold')}"
    )

    headers = {"Content-Type": "application/json"}

    print("-- Endpoint Model List (GET /models) --")
    status, body = http_json(f"{cfg['base_url']}/models", headers, timeout=10)
    if status == 200 and isinstance(body, dict):
        items = body.get("data") if isinstance(body.get("data"), list) else body.get("models")
        if isinstance(items, list):
            served = [
                (m.get("id") or m.get("name") or "")
                for m in items[:10]
                if isinstance(m, dict)
            ]
            print(f"  [PASS] /models (HTTP 200): {', '.join(served) or '<empty>'}")
        else:
            print("  [PASS] /models (HTTP 200)")
    else:
        print(f"  [WARN] /models (HTTP {status}): {str(body)[:160]}")

    models_to_test = [cfg["master_model"]]
    for m in cfg["discussion_models"]:
        if m not in models_to_test:
            models_to_test.append(m)

    print("-- OpenAI-compatible Local Chat Completions --")
    req_timeout = min(max(cfg["timeout"], 20), 60)
    for model in models_to_test:
        prompt_text = apply_thinking_directive("Reply with OK", model=model)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        if cfg["provider"] in LOCAL_LLM_PROVIDERS and cfg.get("thinking") == "false":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        status, body = http_json(
            f"{cfg['base_url']}/chat/completions",
            headers,
            payload=payload,
            timeout=req_timeout,
        )
        if status == 400 and "chat_template_kwargs" in payload:
            payload.pop("chat_template_kwargs", None)
            status, body = http_json(
                f"{cfg['base_url']}/chat/completions",
                headers,
                payload=payload,
                timeout=req_timeout,
            )
        if status == 200 and isinstance(body, dict):
            msg = (body.get("choices") or [{}])[0].get("message") or {}
            raw_content = msg.get("content") or ""
            raw_reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            cleaned = strip_thinking_tags(raw_content).replace("\n", " ")
            if not cleaned and raw_reasoning:
                print(
                    f"  [WARN] {model} (HTTP 200): content is EMPTY while reasoning_content has {len(raw_reasoning)} chars! "
                    "Start llama-server with `--jinja --reasoning off` (or `--reasoning-budget 0`) and set `LLM_THINKING=false` in .env."
                )
            elif not cleaned:
                print(f"  [WARN] {model} (HTTP 200): <empty / thinking only>")
            else:
                suffix = (
                    f" (note: server also emitted {len(raw_reasoning)} reasoning_content chars)"
                    if raw_reasoning
                    else ""
                )
                print(f"  [PASS] {model} (HTTP 200): {cleaned[:80]}{suffix}")
        else:
            print(f"  [FAIL] {model} (HTTP {status}): {str(body)[:160]}")

    if cfg["embedding_model"]:
        print("-- OpenAI-compatible Local Embeddings (Optional) --")
        status, body = http_json(
            f"{cfg['base_url']}/embeddings",
            headers,
            payload={"model": cfg["embedding_model"], "input": "How to compile the project?"},
            timeout=20,
        )
        if status == 200 and isinstance(body, dict) and isinstance(body.get("data"), list) and body["data"]:
            dim = len(body["data"][0].get("embedding", []))
            print(f"  [PASS] {cfg['embedding_model']} (HTTP 200): dim={dim}")
        else:
            print(
                f"  [OPTIONAL] {cfg['embedding_model']} unavailable (HTTP {status}): {str(body)[:120]}"
            )


def main() -> int:
    repo_root = _REPO_ROOT
    load_env_file(os.path.join(repo_root, ".env"))
    check_python_and_packages()
    check_docker()
    check_llm_provider()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
