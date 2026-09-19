# LLM From Scratch — Project Scaffold

Project này được tổ chức theo pipeline: dữ liệu → tokenizer tự viết → decoder LLM LFM-inspired 0.3B tự train → Triton attention → scaling law → SFT/DPO/RLVR → structured decision benchmark → agent context dài → report/demo.

## Mục tiêu học phần và mục tiêu nghiên cứu

Mục tiêu bắt buộc là tự tay viết tokenizer và các thành phần LLM/Transformer, sau đó train một model decoder-only LFM-inspired khoảng 0.3B. Model này là trọng tâm để học data pipeline, pretraining, kernel, alignment và agent; không thay bằng checkpoint có sẵn.

Trên model tự xây, dự án có thêm một nhánh nghiên cứu edge AI:

> Can a 300M LFM-inspired model with parallel calibrated decision heads outperform LFM2-350M and Qwen3-0.6B autoregressive JSON generation in latency, structured-decision accuracy, and calibration for on-device tasks?

Benchmark gồm sáu model:

| Vai trò | Model |
| --- | --- |
| Proposed system | LFM-inspired 300M tự train + parallel `action`, `slot`, `confidence`, `abstain` heads |
| AR baseline gần kiến trúc | LFM2-350M |
| AR Transformer baseline | Qwen3-0.6B |
| Small-model reference | SmolLM2-135M |
| Small-model reference | MobileLLM-125M |
| Small-model reference | Gemma 3 270M |

LFM2-350M và Qwen3-0.6B là hai baseline chính cho claim chính. SmolLM2, MobileLLM và Gemma là baseline tham chiếu cùng dải small language model. Mọi model ngoài model 300M của nhóm đều dùng autoregressive JSON (ưu tiên JSON Schema/grammar-constrained decoding nếu runtime hỗ trợ); chúng không được gọi là parallel decision-head systems.

## Nguyên tắc làm việc

- Tự triển khai các thành phần cốt lõi, không dùng implementation có sẵn cho tokenizer, Transformer và Triton attention kernel.
- Mỗi thí nghiệm phải có config, command đã chạy, commit code, log, checkpoint và kết quả benchmark.
- Không ghi đè dữ liệu thô; mọi bước xử lý dữ liệu tạo ra một phiên bản mới.
- Kết quả không chỉ lưu ở notebook: phải có script có thể chạy lại.

## Các mốc chính

1. M0 — Environment, repository và phân công nhóm.
2. M1 — Dữ liệu sạch và tokenizer tự viết.
3. M2 — Decoder LLM LFM-inspired 0.3B train được.
4. M3 — Triton attention kernel đúng và nhanh hơn baseline ở các workload mục tiêu.
5. M4 — Scaling law và dự đoán khi scale model.
6. M5 — So sánh SFT, DPO và RLVR trên cùng base model.
7. M6 — Structured-decision benchmark: parallel heads so với AR-JSON trên sáu model.
8. M7 — Agent environment, harness, long-context và self-evaluation.
9. M8 — Report NeurIPS, reproducibility package và live demo.

Xem kế hoạch chi tiết tại [`docs/roadmap.md`](docs/roadmap.md).

Danh sách file cần tự tạo theo từng giai đoạn nằm ở [`docs/file_plan.md`](docs/file_plan.md).
