"""Environment and active LLM provider verification script."""

import importlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from Config import resolve_llm_config, strip_thinking_tags

REQUIRED_PACKAGES = [
    "dotenv",
    "openai",
    "langchain",
    "langchain_openai",
    "paramiko",
    "docker",
    "bs4",
    "duckduckgo_search",
    "chromadb",
    "tenacity",
]


def mask_key(value: str) -> str:
    """Return a masked representation of an API key."""
    if not value:
        return "<EMPTY>"
    if len(value) <= 12:
        return "***"
    return f"{value[:8]}...{value[-4:]}"


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
    """Check Python version and required Python dependencies."""
    print("=== 1. Python & Packages ===")
    print(f"Python: {sys.version.splitlines()[0]} ({sys.executable})")
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"  [INSTALLED] {pkg}")
        except ImportError:
            print(f"  [MISSING]   {pkg}")


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


def check_semgrep() -> None:
    """Check Semgrep CLI availability on Windows host."""
    print("\n=== 3. Semgrep CLI ===")
    semgrep_path = shutil.which("semgrep")
    if not semgrep_path:
        print("semgrep: NOT FOUND on Windows host PATH (can run inside Docker Linux container in Stage 3).")
        return
    rc, out, err = run_cmd(["semgrep", "--version"])
    print(f"semgrep ({semgrep_path}) rc={rc}: {out or err}")


def http_json(url: str, headers: dict, payload: dict = None, timeout: int = 20) -> tuple:
    """Send a JSON GET/POST request and return (status_code, parsed_or_text_body)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    method = "POST" if payload is not None else "GET"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
    """Test whichever LLM provider/endpoint is currently active in .env."""
    print("\n=== 4. Active LLM Provider & Endpoint ===")
    try:
        cfg = resolve_llm_config()
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return

    print(f"LLM_PROVIDER:      {cfg['provider']}")
    print(f"LLM_BASE_URL:      {cfg['base_url']}")
    print(f"API_KEY:           {mask_key(cfg['api_key'])}")
    print(f"MASTER_MODEL:      {cfg['master_model']}")
    print(f"DISCUSSION_MODELS: {', '.join(cfg['discussion_models'])}")
    print(f"LLM_TIMEOUT:       {cfg['timeout']}s")

    if cfg["requires_api_key"] and not cfg["api_key"]:
        print(f"[FAIL] API key is empty for LLM_PROVIDER='{cfg['provider']}'.")
        return

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }

    if cfg["provider"] != "google":
        print("-- Endpoint Model List (GET /models) --")
        status, body = http_json(f"{cfg['base_url']}/models", headers, timeout=10)
        if status == 200 and isinstance(body, dict) and isinstance(body.get("data"), list):
            served = [m.get("id", "") for m in body["data"][:10] if isinstance(m, dict)]
            print(f"  [PASS] /models (HTTP 200): {', '.join(served) or '<empty>'}")
        else:
            print(f"  [WARN] /models (HTTP {status}): {str(body)[:160]}")

    models_to_test = [cfg["master_model"]]
    for m in cfg["discussion_models"]:
        if m not in models_to_test:
            models_to_test.append(m)

    print("-- OpenAI-compatible Chat Completions --")
    req_timeout = min(max(cfg["timeout"], 20), 60)
    for model in models_to_test:
        status, body = http_json(
            f"{cfg['base_url']}/chat/completions",
            headers,
            payload={"model": model, "messages": [{"role": "user", "content": "Reply with OK"}]},
            timeout=req_timeout,
        )
        if status == 200 and isinstance(body, dict):
            raw_content = body["choices"][0]["message"]["content"] or ""
            cleaned = strip_thinking_tags(raw_content).replace("\n", " ")
            print(f"  [PASS] {model} (HTTP 200): {cleaned[:80] or '<thinking only>'}")
        else:
            print(f"  [FAIL] {model} (HTTP {status}): {str(body)[:160]}")

    if cfg["provider"] == "google" and cfg["embedding_model"]:
        print("-- OpenAI-compatible Embeddings --")
        status, body = http_json(
            f"{cfg['base_url']}/embeddings",
            headers,
            payload={"model": cfg["embedding_model"], "input": "How to compile the project?"},
            timeout=20,
        )
        if status == 200 and isinstance(body, dict):
            dim = len(body["data"][0]["embedding"])
            print(f"  [PASS] {cfg['embedding_model']} (HTTP 200): dim={dim}")
        else:
            print(f"  [FAIL] {cfg['embedding_model']} (HTTP {status}): {str(body)[:160]}")


def main() -> int:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    load_env_file(os.path.join(repo_root, ".env"))
    check_python_and_packages()
    check_docker()
    check_semgrep()
    check_llm_provider()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
