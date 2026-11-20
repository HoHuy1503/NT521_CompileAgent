# CompileAgent Benchmark & Ablation Evaluation Report

## 1. Full CompileAgent Benchmark Results

| Project | Model Result | Checker (Non-Strict) | Checker (Strict) | Smoke Check (`file`/`ldd`) | Matched Artifacts | Iterations | Time (s) | Shell | Navigator | ErrorSolver | Diagnosis |
| :--- | :-: | :-: | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :--- |
| `7z` | True | False | False | PASS (1) | `-` | 6 | 215.41 | 3 | 1 | 0 | False Positive (Model=True / Checker=False) |
| `curl` | False | True | True | PASS (1) | `lib/.libs/libcurl.so` | 30 | 701.75 | 25 | 1 | 0 | False Negative (Model!=True / Checker=True) |
| `jansson` | True | True | False | PASS (1) | `build/lib/libjansson.a` | 3 | 100.60 | 2 | 1 | 0 | Success |
| `json-c` | True | True | True | PASS (1) | `build/libjson-c.so` | 2 | 64.43 | 2 | 0 | 0 | Success |
| `libbcrypt` | True | True | True | PASS (1) | `bcrypt.o` | 7 | 499.51 | 4 | 1 | 0 | Success |
| `libpcap` | True | True | False | PASS (1) | `libpcap.a` | 8 | 76.40 | 5 | 1 | 1 | Success (Recovered w/ ErrorSolver) |
| `linenoise` | True | False | False | PASS (2) | `-` | 5 | 18.26 | 3 | 1 | 0 | False Positive (Model=True / Checker=False) |
| `quickjs` | True | True | True | PASS (2) | `qjs, qjsc` | 4 | 68.96 | 2 | 1 | 0 | Success |
| `tiny-AES-c` | True | True | False | PASS (1) | `aes.o` | 4 | 8.12 | 3 | 0 | 0 | Success |
| `uhubctl` | True | True | True | PASS (1) | `uhubctl` | 6 | 149.92 | 4 | 1 | 0 | Success |
| `zlib` | True | True | True | PASS (4) | `libz.so, minigzip, minigzip64, minigzipsh` | 10 | 161.42 | 8 | 1 | 0 | Success |
| `zstd` | True | True | True | PASS (3) | `lib/libzstd.a, lib/libzstd.so, lib/obj/conf_f66326bd74ed8f637142a4ff76b688be/static/libzstd.a` | 4 | 125.89 | 3 | 1 | 0 | Success |

### Aggregate Metrics (Full CompileAgent)

| Metric | Value |
| :--- | :--- |
| Total Projects | 12 |
| Model Self-Reported Success Rate | 11/12 (91.7%) |
| Checker CSR (Non-Strict, paper metric) | 10/12 (83.3%) |
| Checker CSR (Strict) | 7/12 (58.3%) |
| Smoke Check Pass Rate (`file` & `ldd`) | 10/12 (83.3%) |
| False Positives (`Model=True, Checker=False`) | 2 |
| False Negatives (`Model!=True, Checker=True`) | 1 |
| Average Compile Time / Total Time | 182.56s / 2190.67s |
| Average Iterations | 7.42 |
| Avg Calls (Shell / CompileNavigator / ErrorSolver) | 5.33 / 0.83 / 0.08 (Total Nav=10, Solver=1) |

## 2. Ablation Study (`Full CompileAgent` vs `Baseline: --disable-navigator --disable-error-solver`)

| Project | Full CSR (Non-Strict / Strict) | Full Iter / Time (s) | Full Nav / Solver Calls | Baseline CSR (Non-Strict / Strict) | Baseline Model | Baseline Iter / Time (s) | Baseline Shell |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `7z` | False / False | 6 / 215.41 | 1 / 0 | True / True | True | 7 / 317.04 | 6 |
| `curl` | True / True | 30 / 701.75 | 1 / 0 | True / True | True | 2 / 197.26 | 2 |
| `libpcap` | True / False | 8 / 76.40 | 1 / 1 | True / True | True | 9 / 73.65 | 5 |
| `quickjs` | True / True | 4 / 68.96 | 1 / 0 | True / True | True | 3 / 64.83 | 3 |
| `uhubctl` | True / True | 6 / 149.92 | 1 / 0 | True / True | True | 7 / 16.15 | 6 |

### Ablation Summary (Matched Subset)

| Configuration | Subset Size | CSR (Non-Strict) | CSR (Strict) | Avg Iterations | Avg Time (s) |
| :--- | :-: | :-: | :-: | :-: | :-: |
| Full CompileAgent | 5 | 4/5 (80.0%) | 3/5 (60.0%) | 10.8 | 242.49 |
| Baseline (No Navigator & No ErrorSolver) | 5 | 5/5 (100.0%) | 5/5 (100.0%) | 5.6 | 133.79 |

CSR Gain (Full vs Baseline, Non-Strict): **-20.0 percentage points**

