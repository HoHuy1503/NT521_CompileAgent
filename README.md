# NT521 CompileAgent (Local LLM Edition)

Hệ thống tự động biên dịch mã nguồn C/C++ từ repository GitHub bằng Multi-Agent trong Docker, triển khai chuẩn xác theo kiến trúc kỹ thuật của bài báo [CompileAgent / AutoCompiler](https://github.com/Ch3nYe/AutoCompiler) (ACL 2025) và chạy **100% bằng Local LLMs** (không sử dụng Cloud API Key).

## Kiến trúc hệ thống

![Kiến trúc hệ thống CompileAgent](docs/images/workflow.png)

Với mỗi repository trong file cấu hình JSON, hệ thống thực hiện theo luồng 3 thành phần cốt lõi của `AutoCompiler`:

1. **`MasterAgent` (`scripts/compile.py`, `src/compile_agent/agent/executor.py`)**:
   - Điều phối quá trình clone code (`downloader.py`) và mở phiên tương tác SSH trong Docker container Linux (`autocompiler:gcc13`).
   - Vận hành vòng lặp ReAct (tối đa 30 bước) kết nối 3 công cụ: `Shell`, `CompileNavigator`, `ErrorSolver`.
   - Tích hợp `SafeReActSingleInputOutputParser`, nén ngữ cảnh scratchpad (`_compact_observation`) và `State-Epoch Repetition Guard` để hoạt động ổn định với Local Reasoning LLMs.
2. **`CompileNavigator` (`src/compile_agent/agent/navigator.py`)**:
   - Đóng vai trò sub-agent chuyên tìm hướng dẫn biên dịch trong mã nguồn repo thông qua 3 công cụ con: `DEBATE`, `SEARCH_INSTRUCTIONS_FROM_FILES`, và `GET_CONTENT_FROM_URL`.
3. **`ErrorSolver` (`src/compile_agent/agent/error_solver.py`)**:
   - Triển khai đúng thuật toán **Multi-Agent Debate 3 tác tử qua 2 vòng** (Phase 1: Round 0 Initial Response + Phase 2: Round 1 Debate) kết hợp **Weighted Confidence Voting** (`trans_confidence`, `count_common_words` với `threshold=30`) và công cụ tìm kiếm `Search_Agent` (`web_search.py`).
   - Khi chạy với 1 model local, 3 tác tử dùng 3 mức nhiệt độ (`temperature = [0.1, 0.4, 0.7]`) để tạo đa dạng góc nhìn; hoặc có thể chỉ định 3 model local khác nhau qua `DISCUSSION_MODELS`.
4. **Kiểm tra & Tổng hợp kết quả (`src/compile_agent/reporting/logs.py`)**:
   - Quét artifact ELF (`\x7fELF`) và static archive (`!<arch>\n`), lọc bỏ file tạm/probe của CMake & Autoconf, đối chiếu với danh sách `files` mục tiêu (Non-Strict & Strict), chạy smoke check (`file`, `ldd`), và lưu log `.txt`/`.json`/`results.jsonl`/`statistics.csv`.

## Cấu trúc thư mục

```text
NT521_CompileAgent/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
├── src/
│   └── compile_agent/
│       ├── __init__.py
│       ├── config.py                  # Cấu hình Local LLM backend, Circuit Breaker, prompt templates
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── executor.py            # ReAct parser/executor, nén scratchpad, chống lặp lệnh
│       │   ├── navigator.py           # CompileNavigator tìm hướng dẫn build trong tài liệu repo
│       │   ├── error_solver.py        # ErrorSolver (3-agent 2-round Debate & Weighted Voting)
│       │   └── web_search.py          # Search_Agent tìm kiếm web bổ trợ khi ErrorSolver cần tra cứu
│       ├── infra/
│       │   ├── __init__.py
│       │   ├── docker_tools.py        # InteractiveDockerShell (SSH PTY) và RAG đọc tài liệu
│       │   └── downloader.py          # Clone repo (git clone --depth 1) và chuẩn hóa LF cho script
│       └── reporting/
│           ├── __init__.py
│           └── logs.py                # Kiểm tra artifact ELF (is_compiled), smoke check, lưu log
├── scripts/
│   ├── compile.py                     # Điểm chạy chính (clone, biên dịch, lưu log)
│   ├── evaluate.py                    # Tổng hợp kết quả benchmark & Ablation ra CSV / Markdown
│   └── check_env.py                   # Kiểm tra Python, Docker và kết nối Local LLM server
├── docker/
│   ├── Dockerfile                     # Định nghĩa image autocompiler:gcc13 (GCC 13.2.0)
│   └── wrappers/
│       ├── wrapper_compiler.sh        # Compiler wrapper (chỉnh cờ biên dịch qua WRAPPER_OPTI)
│       └── wrapper_strip.sh           # Wrapper giữ lại symbol khi strip
├── configs/
│   ├── projects_benchmark.json        # Danh sách 12 repo test nhanh và chạy ablation
│   └── projects.json                  # Danh sách 100 repo benchmark
├── tests/
│   └── test_stability_regression.py   # Bộ unit test kiểm thử hồi quy
├── docs/
│   └── images/
│       └── workflow.png               # Sơ đồ luồng hoạt động hệ thống
├── dataset/                           # Mã nguồn các repo clone về (không track lên Git)
└── outputs/                           # Kết quả chạy và báo cáo (không track lên Git)
```

## Yêu cầu hệ thống

- Windows 10/11 64-bit (hoặc Linux).
- Git.
- Python 3.9 – 3.12.
- Docker Desktop đang chạy ở chế độ Linux containers.
- Server **Local LLM** tương thích OpenAI API (`llama.cpp` / `llama-server`, `Ollama`, `vLLM`, hoặc `LM Studio`) — **không cần API key**.

## Cài đặt môi trường

### Bước 1: Tạo môi trường ảo và cài thư viện Python

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install python-dotenv openai langchain langchain-openai langchain-community langchain-core paramiko docker click beautifulsoup4 tenacity duckduckgo-search googlesearch-python requests chardet pandas tqdm fake-useragent
```

### Bước 2: Build Docker image

Đảm bảo Docker Desktop đang chạy, sau đó build image `autocompiler:gcc13`:

```powershell
docker build -f docker/Dockerfile -t autocompiler:gcc13 .
```

### Bước 3: Cấu hình Local LLM trong `.env`

Tạo `.env` từ file mẫu `.env.example` (không commit `.env` lên Git):

```powershell
copy .env.example .env
```

Mặc định dự án cấu hình chạy với **`llama.cpp` (`llama-server` + Vulkan SDK)** tại `http://localhost:8080/v1`:

```env
LLM_PROVIDER=llamacpp
LLM_BASE_URL=http://localhost:8080/v1
MASTER_MODEL=Qwen 3.8 K_M Q4 27B
DISCUSSION_MODELS=Qwen 3.8 K_M Q4 27B
LLM_CONTEXT_WINDOW=16384
LLM_THINKING=false
```

Khởi chạy `llama-server` với Vulkan SDK (`-ngl 99` đẩy toàn bộ layer lên GPU, `--jinja --reasoning off -c 16384` tắt thinking tránh cắt cụt output ReAct):

```powershell
llama-server -m <path_to_qwen3.8_27b_q4_k_m.gguf> -ngl 99 --port 8080 --jinja --reasoning off -c 16384
```

Nếu dùng **Ollama** hoặc **vLLM**, chỉ cần đổi `LLM_PROVIDER`, `LLM_BASE_URL`, `MASTER_MODEL`, `DISCUSSION_MODELS` trong `.env` theo mẫu có sẵn trong `.env.example`.

### Bước 4: Kiểm tra môi trường

```powershell
python scripts/check_env.py
```

## Hướng dẫn chạy biên dịch & đánh giá

### 1. Chạy thử nhanh 1–2 repo

```powershell
python scripts/compile.py -j configs/projects_benchmark.json -p dataset -l outputs/runs/test_run -c True --projects-filter tiny-AES-c,zlib
```

### 2. Chạy bộ 12 repo (`configs/projects_benchmark.json`)

```powershell
python scripts/compile.py -j configs/projects_benchmark.json -p dataset -l outputs/runs/benchmark_12 -c True --skip-existing
```

### 3. Chạy bộ 100 repo (`configs/projects.json`)

```powershell
python scripts/compile.py -j configs/projects.json -p dataset -l outputs/runs/benchmark_100 -c True --skip-existing
```

### 4. Chạy Ablation Study (tắt `CompileNavigator` và `ErrorSolver`)

```powershell
python scripts/compile.py -j configs/projects_benchmark.json -p dataset -l outputs/runs/ablation_12 -c True --skip-existing --disable-navigator --disable-error-solver
```

### 5. Tổng hợp kết quả (`scripts/evaluate.py`)

So sánh `Full CompileAgent` với `Ablation Baseline`:

```powershell
python scripts/evaluate.py --full-dir outputs/runs/benchmark_12 --ablation-dir outputs/runs/ablation_12 --out-dir outputs/reports/eval_12
```

Hoặc tổng hợp 1 thư mục chạy trên Windows PowerShell:

```powershell
python scripts/evaluate.py --full-dir outputs/runs/benchmark_100 --ablation-dir="" --out-dir outputs/reports/eval_100
```

## Kiểm thử hồi quy (Unit Tests)

```powershell
python -m unittest discover -s tests -v
```

## Tham khảo

- Bài báo ACL 2025: [CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](https://aclanthology.org/2025.acl-long.103/)
- Repository gốc: [Ch3nYe/AutoCompiler](https://github.com/Ch3nYe/AutoCompiler)
