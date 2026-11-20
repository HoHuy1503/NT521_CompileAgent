# CompileAgent Benchmark & Ablation Evaluation Report

## 1. Full CompileAgent Benchmark Results

| Project | Model Result | Checker (Non-Strict) | Checker (Strict) | Smoke Check (`file`/`ldd`) | Matched Artifacts | Iterations | Time (s) | Shell | Navigator | ErrorSolver | Diagnosis |
| :--- | :-: | :-: | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :--- |
| `aubio` | True | True | True | PASS (2) | `build/src/libaubio.so, build/src/libaubio.so.5.4.8` | 11 | 637.97 | 4 | 1 | 0 | Success |
| `boringssl` | UNCERTAIN | False | False | FAIL (0) | `-` | 4 | 274.34 | 0 | 0 | 0 | Compilation Failed |
| `botan` | UNCERTAIN | False | False | FAIL (0) | `-` | 4 | 161.52 | 0 | 0 | 0 | Compilation Failed |
| `cryptopp` | UNCERTAIN | False | False | FAIL (0) | `-` | 5 | 129.12 | 1 | 0 | 0 | Compilation Failed |
| `curl` | UNCERTAIN | False | False | FAIL (0) | `-` | 4 | 176.25 | 0 | 0 | 0 | Compilation Failed |
| `GmSSL` | UNCERTAIN | False | False | FAIL (0) | `-` | 6 | 227.54 | 1 | 1 | 0 | Compilation Failed |
| `libbcrypt` | UNCERTAIN | False | False | FAIL (0) | `-` | 4 | 97.90 | 0 | 0 | 0 | Compilation Failed |
| `LibreSSL` | UNCERTAIN | False | False | FAIL (0) | `-` | 6 | 524.61 | 1 | 1 | 0 | Compilation Failed |
| `libsodium` | UNCERTAIN | False | False | FAIL (0) | `-` | 6 | 216.19 | 1 | 1 | 0 | Compilation Failed |
| `libtomcrypt` | UNCERTAIN | False | False | FAIL (0) | `-` | 7 | 422.26 | 1 | 1 | 0 | Compilation Failed |
| `llama2.c` | UNCERTAIN | False | False | FAIL (0) | `-` | 6 | 443.63 | 1 | 0 | 0 | Compilation Failed |
| `masscan` | UNCERTAIN | False | False | FAIL (0) | `-` | 8 | 832.55 | 1 | 1 | 0 | Compilation Failed |
| `mbedtls` | False | False | False | FAIL (0) | `-` | 8 | 727.11 | 4 | 1 | 0 | Compilation Failed |
| `nettle` | UNCERTAIN | False | False | FAIL (0) | `-` | 9 | 424.27 | 3 | 1 | 0 | Compilation Failed |
| `openssl` | UNCERTAIN | True | True | PASS (4) | `libcrypto.so, libcrypto.so.4, libssl.so, libssl.so.4` | 16 | 1696.64 | 5 | 1 | 0 | False Negative (Model!=True / Checker=True) |
| `redis` | UNCERTAIN | False | False | FAIL (0) | `-` | 4 | 135.79 | 0 | 0 | 0 | Compilation Failed |
| `tea-c` | UNCERTAIN | False | False | FAIL (0) | `-` | 5 | 43.05 | 1 | 0 | 0 | Compilation Failed |
| `tiny-AES-c` | UNCERTAIN | False | False | FAIL (0) | `-` | 5 | 118.76 | 1 | 0 | 0 | Compilation Failed |
| `Tongsuo` | UNCERTAIN | False | False | FAIL (0) | `-` | 3 | 349.94 | 1 | 1 | 0 | Compilation Failed |
| `whisper.cpp` | UNCERTAIN | False | False | FAIL (0) | `-` | 8 | 516.62 | 1 | 1 | 0 | Compilation Failed |
| `wolfssl` | UNCERTAIN | False | False | FAIL (0) | `-` | 4 | 161.23 | 0 | 0 | 0 | Compilation Failed |
| `zstd` | UNCERTAIN | False | False | FAIL (0) | `-` | 4 | 79.13 | 0 | 0 | 0 | Compilation Failed |

### Aggregate Metrics (Full CompileAgent)

| Metric | Value |
| :--- | :--- |
| Total Projects | 22 |
| Model Self-Reported Success Rate | 1/22 (4.5%) |
| Checker CSR (Non-Strict, paper metric) | 2/22 (9.1%) |
| Checker CSR (Strict) | 2/22 (9.1%) |
| Smoke Check Pass Rate (`file` & `ldd`) | 2/22 (9.1%) |
| False Positives (`Model=True, Checker=False`) | 0 |
| False Negatives (`Model!=True, Checker=True`) | 1 |
| Average Compile Time / Total Time | 381.66s / 8396.42s |
| Average Iterations | 6.23 |
| Avg Calls (Shell / CompileNavigator / ErrorSolver) | 1.23 / 0.5 / 0 (Total Nav=11, Solver=0) |
