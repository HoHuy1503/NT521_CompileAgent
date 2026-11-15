# NT521 CompileAgent

Hệ thống tự động biên dịch mã nguồn C/C++ từ repository GitHub bằng Multi-Agent trong Docker, kết hợp pipeline kiểm thử bảo mật 3 lớp gắn với quá trình biên dịch. Dự án được phát triển dựa trên [CompileAgent / AutoCompiler](https://github.com/Ch3nYe/AutoCompiler) (ACL 2025).

## Hệ thống làm gì?

![Kiến trúc hệ thống CompileAgent](docs/images/workflow.png)

Với mỗi repository trong file cấu hình JSON, hệ thống thực hiện theo 2 pha:

1. **Biên dịch tự động (Multi-Agent trong Docker `autocompiler:gcc13`)**:
   - `MasterAgent` điều phối quá trình clone code và build trong container Linux.
   - `CompileNavigator` tìm hướng dẫn biên dịch từ tài liệu trong repo (`README`, `BUILD`, `INSTALL`, ...).
   - `ErrorSolver` phân tích log lỗi khi build thất bại để đề xuất lệnh sửa (cài thư viện thiếu, chỉnh tham số build...).
   - Sau khi agent kết thúc, hệ thống kiểm tra artifact ELF theo `files` mục tiêu và chạy smoke check (`file`, `ldd`).

2. **Pipeline bảo mật 3 lớp (`src/compile_agent/security/audit.py`)**:
   - **Lớp 1 — SAST (Static Analysis)**: Chạy `Semgrep` (bộ luật tùy chỉnh `rules/c_security.yaml` + `p/c`) và `Cppcheck` dùng `compile_commands.json` sinh từ CMake, Ninja hoặc `bear`.
   - **Lớp 2 — Audit cờ bảo vệ ELF**: Dùng `readelf` kiểm tra `NX`, `Stack Canary`, `PIE/DSO`, `RELRO`, `FORTIFY_SOURCE` trên file biên dịch mặc định và chấm điểm 0–100.
   - **Lớp 3 — Hardening Rebuild & Dynamic Sanitizer**:
     - Biên dịch lại với cờ bảo vệ chuẩn (`-O2 -g -fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE -pie -Wl,-z,relro,-z,now`), đo lại điểm bảo vệ Before/After và chạy smoke check.
     - Biên dịch lại với `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer` rồi chạy test suite / nhị phân để phát hiện lỗi bộ nhớ runtime (ASan/UBSan).

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
│       ├── config.py                  # Đọc .env, cấu hình LLM provider, key pool, circuit breaker, prompt
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── executor.py            # ReAct parser/executor, nén scratchpad, chống lặp lệnh
│       │   ├── navigator.py           # CompileNavigator tìm hướng dẫn build trong tài liệu repo
│       │   ├── error_solver.py        # ErrorSolver chẩn đoán lỗi biên dịch
│       │   └── web_search.py          # Tìm kiếm web bổ trợ khi ErrorSolver cần tra cứu thêm
│       ├── infra/
│       │   ├── __init__.py
│       │   ├── docker_tools.py        # Phiên SSH Docker (Shell) và công cụ đọc tài liệu / RAG
│       │   └── downloader.py          # Clone repo (git clone --depth 1) và chuẩn hóa LF cho script build
│       ├── security/
│       │   ├── __init__.py
│       │   └── audit.py               # Pipeline bảo mật 3 lớp (SAST, Checksec, Hardening, ASan/UBSan)
│       └── reporting/
│           ├── __init__.py
│           └── logs.py                # Tìm artifact ELF, smoke check (file/ldd) và lưu log JSON/TXT
├── scripts/
│   ├── compile.py                     # Điểm chạy chính (clone, biên dịch, Stage 3, lưu log)
│   ├── evaluate.py                    # Tổng hợp kết quả benchmark & bảo mật ra CSV / Markdown
│   └── check_env.py                   # Kiểm tra Python, Docker, Semgrep và kết nối LLM
├── docker/
│   ├── Dockerfile                     # Định nghĩa image autocompiler:gcc13
│   └── wrappers/
│       ├── wrapper_compiler.sh        # Wrapper chèn cờ hardening / sanitizer khi build lại
│       └── wrapper_strip.sh           # Wrapper giữ lại symbol debug khi kiểm thử bảo mật
├── rules/
│   └── c_security.yaml                # 5 rule Semgrep tùy chỉnh cho C/C++
├── configs/
│   ├── projects_benchmark.json        # Danh sách 12 repo test nhanh và chạy ablation
│   ├── projects.json                  # Danh sách 100 repo benchmark
│   └── test_stage3.json               # Cấu hình test pipeline bảo mật Stage 3
├── tests/
│   └── fixtures/
│       ├── vuln-c-demo/               # Project C mẫu chứa lỗi bộ nhớ để test Stage 3
│       └── semgrep/
│           └── c_security.c           # File test cho rule Semgrep c_security.yaml
├── docs/
│   └── images/
│       └── workflow.png               # Sơ đồ luồng hoạt động hệ thống
├── dataset/                           # Mã nguồn các repo clone về (không track lên Git)
└── outputs/                           # Kết quả chạy và báo cáo (không track lên Git)
    ├── runs/                          # Log từng lần chạy (JSON, TXT, SAST, security, discussion.log)
    └── reports/                       # Báo cáo CSV / Markdown do scripts/evaluate.py xuất ra
```

## Yêu cầu hệ thống

- Windows 10/11 64-bit (hoặc Linux).
- Git.
- Python 3.9 – 3.12.
- Docker Desktop đang chạy ở chế độ Linux containers.
- Kết nối LLM qua Google Gemini API hoặc server LLM local tương thích OpenAI (Ollama, vLLM, LM Studio).

## Cài đặt môi trường

### Bước 1: Tạo môi trường ảo và cài thư viện Python

Mở terminal (PowerShell / CMD) tại thư mục dự án:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

(Trên Linux dùng `source .venv/bin/activate`.)

Cài thư viện theo phiên bản Python bạn đang dùng:

- **Python 3.10 – 3.12+** (khuyên dùng, cài các gói lõi thực tế để tránh lỗi build wheel của gói cũ):
  ```powershell
  pip install python-dotenv openai langchain langchain-openai langchain-community langchain-core paramiko docker click beautifulsoup4 tenacity duckduckgo-search googlesearch-python requests chardet pandas tqdm fake-useragent
  ```
- **Python 3.9**:
  ```powershell
  pip install -r requirements.txt
  ```

### Bước 2: Build Docker image

Đảm bảo Docker Desktop đang chạy, sau đó build image `autocompiler:gcc13` (đã cài sẵn GCC 13.2.0, CMake, Bear, Cppcheck, GDB, Semgrep, bộ luật `/opt/rules` và compiler wrapper):

```powershell
docker build -f docker/Dockerfile -t autocompiler:gcc13 .
```

### Bước 3: Tạo file `.env`

Tạo `.env` từ file mẫu `.env.example` (tuyệt đối không commit `.env` lên Git):

```powershell
copy .env.example .env
```

(Trên Linux: `cp .env.example .env`)

Mở `.env` và điền một trong hai cấu hình phổ biến dưới đây:

- **Cách A — Dùng Google Gemini API (mặc định)**:
  ```env
  LLM_PROVIDER=google
  GOOGLE_API_KEY=your_google_api_key_here
  ```
- **Cách B — Dùng model local qua Ollama (ví dụ `qwen3:30b`)**:
  ```env
  LLM_PROVIDER=ollama
  LLM_BASE_URL=http://localhost:11434/v1
  MASTER_MODEL=qwen3:30b
  DISCUSSION_MODELS=qwen3:30b
  ```
  (Trong `.env.example` có sẵn mẫu cho `vllm`, `lmstudio`, `openai_compatible`. Có thể chỉnh `MAKE_JOBS=4` tùy số luồng CPU của máy.)

### Bước 4: Kiểm tra môi trường

```powershell
python scripts/check_env.py
```

Khi các mục kiểm tra package Python, Docker daemon, image `autocompiler:gcc13`, Semgrep và kết nối LLM đều báo OK là hệ thống sẵn sàng chạy.

## Hướng dẫn chạy biên dịch & kiểm thử bảo mật

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

Lưu ý khi chạy lô lớn:
- `--skip-existing` giúp bỏ qua các repo đã có log hợp lệ trong thư mục `--log_path`. Nếu bị ngắt giữa chừng (hết quota ngày, tắt máy, rớt mạng LLM...), chỉ cần chạy lại đúng lệnh cũ để tiếp tục phần còn lại.
- Mặc định hệ thống chạy tuần tự để tránh nghẽn VRAM model local hoặc vượt giới hạn RPM của API. Nếu hạ tầng đủ mạnh, có thể bật chạy song song bằng `--multi_process True --max-workers 2`.

### 4. Chạy Ablation (tắt `CompileNavigator` và `ErrorSolver`)

```powershell
python scripts/compile.py -j configs/projects_benchmark.json -p dataset -l outputs/runs/ablation_12 -c True --skip-existing --disable-navigator --disable-error-solver --security-mode none
```

### Tóm tắt các tham số CLI của `scripts/compile.py`

- `-j, --json_path`: File JSON danh sách repo (bắt buộc).
- `-p, --dataset_base_path`: Thư mục lưu repo clone về, ví dụ `dataset` (bắt buộc).
- `-l, --log_path`: Thư mục lưu log và báo cáo của lần chạy (bắt buộc).
- `-c, --clean_copied_project`: `True` để xóa bản copy tạm sau khi chạy xong repo, `False` để giữ lại khi debug (bắt buộc).
- `--skip-existing`: Bỏ qua repo đã có log hợp lệ trong `--log_path`.
- `--projects-filter`: Lọc chạy theo danh sách tên repo, cách nhau bằng dấu phẩy.
- `--security-mode`: Chọn phạm vi pipeline bảo mật Stage 3: `all` (mặc định), `sast`, `hardening`, `sanitizer`, hoặc `none`.
- `--disable-navigator` / `--disable-error-solver`: Tắt tool điều hướng hoặc tool sửa lỗi cho thử nghiệm ablation.
- `-r, --retry`: Số lần thử lại khi checker chưa pass (mặc định `0`).
- `-s, --strict_checker`: `True` yêu cầu khớp toàn bộ file trong `files`; `False` (mặc định) chỉ cần khớp ít nhất 1 file.
- `--multi_process` / `--max-workers`: Bật xử lý song song và số worker tối đa (mặc định `False`, `max-workers=2`).

## Tổng hợp kết quả (`scripts/evaluate.py`)

Sau khi chạy xong `scripts/compile.py`, dùng `scripts/evaluate.py` để xuất bảng CSV và báo cáo Markdown.

**Tổng hợp 1 thư mục chạy (kèm bảng bảo mật Stage 3):**

- Trên **Windows PowerShell** (dùng dấu `=` cho giá trị rỗng):
  ```powershell
  python scripts/evaluate.py --full-dir outputs/runs/benchmark_100 --ablation-dir="" --out-dir outputs/reports/eval_100
  ```
- Trên **CMD hoặc Linux bash**:
  ```bash
  python scripts/evaluate.py --full-dir outputs/runs/benchmark_100 --ablation-dir "" --out-dir outputs/reports/eval_100
  ```

**So sánh Full CompileAgent với Ablation Baseline:**

```powershell
python scripts/evaluate.py --full-dir outputs/runs/benchmark_12 --ablation-dir outputs/runs/ablation_12 --out-dir outputs/reports/eval_12
```

## File kết quả đầu ra

Trong thư mục `--log_path` (ví dụ `outputs/runs/benchmark_100/`):
- `<project>.log.<timestamp>.json` và `.txt`: Log hội thoại của agent, `model_result`, `checker_result`, smoke check và tóm tắt bảo mật.
- `semgrep/<project>.semgrep.json`: Kết quả quét Semgrep.
- `cppcheck/<project>.cppcheck.json`: Kết quả quét Cppcheck.
- `security/<project>.security.json`: Báo cáo chi tiết cả 3 lớp bảo mật của từng repo.
- `statistics.csv`: Bảng thống kê thô từng lượt chạy.
- `download_errors.jsonl`: Danh sách repo lỗi khi `git clone` hoặc copy (không làm sập cả batch).

Trong thư mục `--out-dir` do `scripts/evaluate.py` tạo (ví dụ `outputs/reports/eval_100/`):
- `benchmark_details.csv`: Bảng kết quả biên dịch chi tiết từng repo (tự lọc bản ghi mới nhất nếu chạy nhiều lần).
- `security_summary.csv`: Bảng tổng hợp chỉ số SAST, điểm Checksec Before/After, kết quả Hardening và ASan/UBSan từng repo.
- `ablation_comparison.csv`: Bảng so sánh Full vs Baseline (khi có `--ablation-dir`).
- `benchmark_report.md`: Báo cáo tổng hợp dạng Markdown.

## Tham khảo

- Bài báo ACL 2025: [CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](https://aclanthology.org/2025.acl-long.103/)
- Repository gốc: [Ch3nYe/AutoCompiler](https://github.com/Ch3nYe/AutoCompiler)

```bibtex
@inproceedings{hu2025compileagent,
  title={CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System},
  author={Hu, Li and Chen, Guoqiang and Shang, Xiuwei and Cheng, Shaoyin and Wu, Benlong and Li, Gangyang and Zhu, Xu and Zhang, Weiming and Yu, Nenghai},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics},
  year={2025},
  url={https://aclanthology.org/2025.acl-long.103/}
}
```
