# Measuring LLM Throughput (Tokens per Second) in OpenVINO GenAI

## Overview

OpenVINO GenAI does not currently provide a single built-in API that directly
returns token throughput (tokens per second) for large language model (LLM)
inference.

This is intentional, as throughput depends on multiple factors such as:
- Tokenization strategy
- Streaming vs. non-streaming inference
- Batch size and scheduling
- Model architecture and decoding strategy

Instead, OpenVINO provides lower-level primitives and utilities that allow
applications to accurately measure these metrics at the application level.

---

## Recommended Approach for GenAI Pipelines

For GenAI workflows, the recommended approach to measure throughput is to use
**streaming inference callbacks** and capture timing information while tokens
are being generated.

By counting the number of tokens produced over a measured time window,
applications can compute:
- Tokens per second (throughput)
- First-token latency
- Second-token latency (often used in LLM benchmarks)

This approach aligns with how OpenVINO benchmark tools report LLM performance,
while remaining flexible for different application requirements.

---

## Example: Measuring Tokens per Second (Python)

The following example demonstrates how to measure token throughput using
a streaming callback in an OpenVINO GenAI pipeline.

```python
import time
from openvino_genai import LLMPipeline

pipeline = LLMPipeline(model_path="model", device="CPU")

token_count = 0
start_time = None
second_token_time = None

def on_token(token):
    global token_count, start_time, second_token_time

    current_time = time.perf_counter()

    if token_count == 0:
        start_time = current_time
    elif token_count == 1:
        second_token_time = current_time

    token_count += 1

pipeline.generate(
    prompt="Explain transformers in simple terms",
    streaming_callback=on_token
)

end_time = time.perf_counter()

total_time = end_time - start_time
tokens_per_second = token_count / total_time

print(f"Tokens generated: {token_count}")
print(f"Tokens per second: {tokens_per_second:.2f}")
print(f"Second-token latency: {second_token_time - start_time:.4f}s")


