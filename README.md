# CompileAgent và đánh giá bảo mật mã nguồn

> Đồ án này dựa trên [CompileAgent](https://github.com/Ch3nYe/AutoCompiler), một hệ thống dùng LLM và công cụ tích hợp để tự động tìm cách biên dịch các repository thực tế. Nhóm mở rộng quy trình bằng bước quét bảo mật tĩnh Semgrep sau khi agent hoàn tất.

## Mục tiêu

- Nhận repository và thử biên dịch trong container Docker.
- Dùng LLM để tìm hướng dẫn build, thực thi lệnh và xử lý lỗi.
- Chạy Semgrep với bộ luật `p/security-audit` trên mã nguồn dự án.
- Lưu log hội thoại/build và báo cáo Semgrep JSON để phục vụ đánh giá.

Semgrep là công cụ phân tích tĩnh. Kết quả của nó là cảnh báo cần xem xét, không tự động đồng nghĩa với lỗ hổng đã xác nhận.

## Thành phần

- `CompileAgent.py`: điều phối agent biên dịch và gọi Semgrep.
- `Tools.py`, `MultiAgentGetInstructions.py`, `MultiAgentDiscussion.py`: công cụ hỗ trợ agent.
- `Logs.py`: lưu log, kết quả build và tóm tắt bảo mật.
- `Config.py`: cấu hình model, tracing và đường dẫn log; khóa API được đọc từ biến môi trường.
- `Dockerfile`: môi trường build Linux dùng image `autocompiler:gcc13`.

## Yêu cầu

- Windows 10/11 64-bit.
- Python 3.9 và Git.
- Docker Desktop chạy Linux containers.
- Khóa API tương thích OpenAI; cấu hình hiện tại dùng OpenRouter.
- Kết nối mạng để tải repository và bộ luật Semgrep.

> Môi trường này được thử trên Windows với Python 3.9. Trong quá trình thiết lập, `uvloop` được bỏ qua vì không hỗ trợ Windows; `readmeai` được bỏ qua do xung đột phiên bản `tiktoken` và không cần cho luồng thử nghiệm hiện tại. Các thay đổi tương ứng nằm trong `Requirements.txt`.

## Cài đặt

Mở CMD tại thư mục dự án:

```bat
py -3.9 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r Requirements.txt
pip install semgrep
```

Build image Docker:

```bat
docker build -t autocompiler:gcc13 .
```

Đợi Docker Desktop báo đang chạy trước khi khởi động agent.

## Cấu hình API

Không ghi API key trực tiếp vào `Config.py`, README, JSON mẫu hoặc Git. `Config.py` đọc các biến `OPENAI_API_KEY` và `DEEPSEEK_API_KEY` từ môi trường. Có thể dùng cùng một khóa OpenRouter cho hai biến trong cửa sổ CMD hiện tại:

```bat
set OPENAI_API_KEY=YOUR_OPENROUTER_KEY
set DEEPSEEK_API_KEY=YOUR_OPENROUTER_KEY
```

Thay `YOUR_OPENROUTER_KEY` trong máy của bạn; không commit khóa. `OPENAI_BASE_URL`, `DEEPSEEK_BASE_URL`, tên model và model embedding được khai báo trong `Config.py` và có thể đổi theo model/API bạn dùng.

## Chạy thử một dự án

Tạo `one_project.json` ở thư mục gốc (file này nằm trong `.gitignore` vì cấu hình thử nghiệm là cục bộ):

```json
[
  {
    "name": "tiny-AES-c",
    "url": "https://github.com/kokke/tiny-AES-c",
    "files": ["aes.o", "aes.c.o"]
  }
]
```

Tạo thư mục log và chạy:

```bat
if not exist logs mkdir logs
python CompileAgent.py -j one_project.json -p Dataset -l logs -c False
```

`-c False` là tùy chọn CLI của dự án để không xóa bản sao sau khi chạy; kiểm tra `python CompileAgent.py --help` để xem các tùy chọn khác. Agent tải hoặc dùng mã nguồn trong `Dataset`, sau đó làm việc trên bản sao trong container.

## Quét bảo mật riêng

Có thể quét thủ công mã nguồn và lưu báo cáo JSON:

```bat
semgrep scan --config p/security-audit --json --output logs\tiny-AES-c-semgrep.json Dataset\tiny-AES-c
```

Trong luồng tích hợp, `CompileAgent.py` gọi Semgrep sau bước agent và tạo `logs\<project>-semgrep.json`. Log mỗi lượt (`.txt` và `.json`) chứa kết quả agent, checker và tóm tắt Semgrep gồm trạng thái, tổng phát hiện, phân bố mức độ và đường dẫn báo cáo.

Các cảnh báo cần được diễn giải theo phạm vi ruleset và xác minh trước khi gọi là lỗ hổng. Báo cáo JSON cũng ghi danh sách file được quét và lỗi nếu có.

## Kết quả và giới hạn hiện biết

- Log AutoCompiler được lưu dưới `logs/`; thư mục này không được commit.
- Tệp `*-semgrep.json` là báo cáo chi tiết của Semgrep.
- `model_result` là kết luận do agent đưa ra; `checker_result` là kiểm tra artifact theo danh sách `files` trong JSON đầu vào. Hai giá trị có thể khác nhau nếu tên/định dạng artifact không khớp.
- Luồng Docker dùng SSH vào container và có phần tương thích Windows/Docker Desktop trong `Tools.py`.
- Semgrep cần mạng để lấy ruleset registry; nếu không kết nối được, báo cáo có thể không được tạo và log sẽ ghi lỗi.

## Nguồn tham khảo

- Bài báo ACL 2025: [CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](https://aclanthology.org/2025.acl-long.103/).
- Repository gốc: [Ch3nYe/AutoCompiler](https://github.com/Ch3nYe/AutoCompiler).

## Trích dẫn bài báo

```bibtex
@inproceedings{hu2025compileagent,
  title={CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System},
  author={Hu, Li and Chen, Guoqiang and Shang, Xiuwei and Cheng, Shaoyin and Wu, Benlong and Li, Gangyang and Zhu, Xu and Zhang, Weiming and Yu, Nenghai},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics},
  year={2025},
  url={https://aclanthology.org/2025.acl-long.103/}
}
```
