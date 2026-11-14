import logging
import os
import re
import time
from dotenv import load_dotenv

ENV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_FILE_PATH)


def call_with_rate_limit_retry(fn, *args, max_attempts=4, **kwargs):
    """Execute fn with smart backoff on per-minute 429 RPM limits and transient timeout/5xx errors."""
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            err_text = str(exc)
            err_name = type(exc).__name__
            if "PerDay" in err_text or "GenerateRequestsPerDay" in err_text:
                raise
            is_rate_limit = (
                "429" in err_text
                or "RESOURCE_EXHAUSTED" in err_text
                or "RateLimitError" in err_name
            )
            is_transient = (
                "Request timed out" in err_text
                or "timed out" in err_text.lower()
                or "APITimeoutError" in err_name
                or "APIConnectionError" in err_name
                or "Connection error" in err_text
                or "502" in err_text
                or "503" in err_text
                or "InternalServerError" in err_name
            )
            if (not is_rate_limit and not is_transient) or attempt == max_attempts - 1:
                raise
            if is_rate_limit:
                wait_seconds = 22.0
                m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_text, re.IGNORECASE)
                if not m:
                    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", err_text)
                if m:
                    wait_seconds = min(max(float(m.group(1)) + 2.0, 5.0), 65.0)
                logging.warning(
                    f"[!] Hit API RPM rate limit (attempt {attempt + 1}/{max_attempts}); sleeping {wait_seconds:.1f}s..."
                )
            else:
                wait_seconds = 5.0
                logging.warning(
                    f"[!] Transient API error ({err_name}: {err_text[:120]}) (attempt {attempt + 1}/{max_attempts}); sleeping {wait_seconds:.1f}s..."
                )
            time.sleep(wait_seconds)


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
        "master_model": "gemini-3.5-flash-lite",
        "discussion_models": [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.1-flash-lite-preview",
        ],
        "embedding_model": "gemini-embedding-001",
        "timeout": 60,
        "api_key_fallback": "",
        "requires_api_key": True,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "master_model": "qwen3:30b",
        "discussion_models": None,
        "embedding_model": "nomic-embed-text",
        "timeout": 180,
        "api_key_fallback": "ollama",
        "requires_api_key": False,
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "master_model": "Qwen/Qwen3-30B-A3B",
        "discussion_models": None,
        "embedding_model": "BAAI/bge-m3",
        "timeout": 180,
        "api_key_fallback": "EMPTY",
        "requires_api_key": False,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "master_model": "qwen/qwen3-30b-a3b",
        "discussion_models": None,
        "embedding_model": "text-embedding-nomic-embed-text-v1.5",
        "timeout": 180,
        "api_key_fallback": "lm-studio",
        "requires_api_key": False,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "master_model": "gpt-4o-mini",
        "discussion_models": ["gpt-4o-mini"],
        "embedding_model": "text-embedding-3-small",
        "timeout": 60,
        "api_key_fallback": "",
        "requires_api_key": True,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "master_model": "qwen/qwen3-30b-a3b",
        "discussion_models": None,
        "embedding_model": "openai/text-embedding-3-small",
        "timeout": 90,
        "api_key_fallback": "",
        "requires_api_key": True,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "master_model": "deepseek-chat",
        "discussion_models": ["deepseek-chat"],
        "embedding_model": "",
        "timeout": 90,
        "api_key_fallback": "",
        "requires_api_key": True,
    },
    "openai_compatible": {
        "base_url": "http://localhost:8000/v1",
        "master_model": "qwen3-30b",
        "discussion_models": None,
        "embedding_model": "",
        "timeout": 180,
        "api_key_fallback": "EMPTY",
        "requires_api_key": False,
    },
}

_PROVIDER_ALIASES = {
    "gemini": "google",
    "local": "ollama",
    "custom": "openai_compatible",
    "openai-compatible": "openai_compatible",
}


def resolve_llm_config(provider: str = None) -> dict:
    """Resolve active LLM endpoint, credentials, models, and timeout from env."""
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

    explicit_key = os.getenv("LLM_API_KEY", "").strip()
    if canonical == "google":
        api_key = explicit_key or os.getenv("GOOGLE_API_KEY", "").strip()
    elif canonical == "openai":
        api_key = explicit_key or os.getenv("OPENAI_API_KEY", "").strip()
    elif canonical == "deepseek":
        api_key = explicit_key or os.getenv("DEEPSEEK_API_KEY", "").strip()
    elif canonical == "openrouter":
        api_key = explicit_key or os.getenv("OPENROUTER_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    elif canonical in ("vllm", "openai_compatible"):
        api_key = explicit_key or os.getenv("OPENAI_API_KEY", "").strip() or preset["api_key_fallback"]
    else:
        api_key = explicit_key or preset["api_key_fallback"]

    master_model = os.getenv("MASTER_MODEL", "").strip() or preset["master_model"]
    discussion_env = os.getenv("DISCUSSION_MODELS", "").strip()
    if discussion_env:
        discussion_models = [m.strip() for m in discussion_env.split(",") if m.strip()]
    elif preset["discussion_models"]:
        discussion_models = list(preset["discussion_models"])
    else:
        # Single-model default avoids forcing local servers to load 3 separate models into VRAM
        discussion_models = [master_model]

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip() or preset["embedding_model"]
    timeout_env = os.getenv("LLM_TIMEOUT", "").strip()
    timeout = int(timeout_env) if timeout_env else preset["timeout"]

    return {
        "provider": canonical,
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "master_model": master_model,
        "discussion_models": discussion_models,
        "embedding_model": embedding_model,
        "timeout": timeout,
        "requires_api_key": preset["requires_api_key"],
    }


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by models such as Qwen3 or DeepSeek-R1."""
    if not text:
        return ""
    raw = str(text)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    if "<think>" not in raw.lower() and "</think>" in cleaned.lower():
        cleaned = re.sub(
            r"^(?!.*Action\s*Input\s*:).*?</think>",
            "",
            cleaned,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        )
    cleaned = re.sub(r"^\s*<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def create_chat_llm(
    model: str = None,
    temperature: float = None,
    timeout: int = None,
    max_retries: int = 1,
    base_url: str = None,
    api_key: str = None,
):
    """Instantiate a ChatOpenAI client bound to the active provider configuration."""
    from langchain_openai import ChatOpenAI

    cfg = resolve_llm_config()
    resolved_base_url = (base_url or cfg["base_url"]).rstrip("/")
    resolved_api_key = cfg["api_key"] if api_key is None else api_key
    if not resolved_api_key:
        raise ValueError(
            f"Missing API key for LLM_PROVIDER='{cfg['provider']}'. "
            f"Set LLM_API_KEY (or GOOGLE_API_KEY for google) in {ENV_FILE_PATH}."
        )
    resolved_model = model or cfg["master_model"]
    resolved_temp = LLM_TEMPERATURE if temperature is None else temperature
    resolved_timeout = cfg["timeout"] if timeout is None else timeout
    return ChatOpenAI(
        base_url=resolved_base_url,
        model=resolved_model,
        api_key=resolved_api_key,
        temperature=resolved_temp,
        timeout=resolved_timeout,
        max_retries=max_retries,
    )


_ACTIVE_LLM_CONFIG = resolve_llm_config()
LLM_PROVIDER = _ACTIVE_LLM_CONFIG["provider"]
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
LLM_BASE_URL = _ACTIVE_LLM_CONFIG["base_url"]
LLM_API_KEY = _ACTIVE_LLM_CONFIG["api_key"]
GEMINI_BASE_URL = LLM_BASE_URL
MASTER_MODEL = _ACTIVE_LLM_CONFIG["master_model"]
DISCUSSION_MODELS = _ACTIVE_LLM_CONFIG["discussion_models"]

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT = _ACTIVE_LLM_CONFIG["timeout"]
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
PROCESS_LOG_CSV_PATH = "logs"
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
        return model_decision_template.format(project_name=project_name)

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
    return (
        f"I want you to help me compile the project {project_name} from source code. Please complete the compilation task step by step.\n\n"
        f"The project has been downloaded into /work/, which is also the Shell's initial path.\n"
        f"You should follow these rules:\n{rules_block}\n"
    )


discussion_template1 = """You are an experienced compiler troubleshooting expert. Your task is to analyze provided error messages, identify likely causes, and deliver specific solutions. 

The entire compilation process was performed on Debian GNU/Linux 12 (bookworm) with GCC 13.2.0 and root user. Note that APT package indices are NOT cached in this container, so any apt installation command must include `apt-get update && apt-get install -y <pkg>`.
the error messages during {project} compilation are as follows:
{input}

Unless you really don't know how to solve this problem, there is no need to call the following tools:
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

Unless you really don't know how to solve this problem, there is no need to call the following tools:
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
