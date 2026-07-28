# INT4 KV Cache Compression for LLM Inference on Intel GPU: New in OpenVINO 2026.2

*By Mingyu Kim and Byungil Min | June 1, 2026*

Running large language models at long context lengths is memory-intensive work. Even after compressing model weights to INT4, the **KV cache** keeps growing with every new token and every prompt you push through the model. OpenVINO 2026.2 introduces INT4 KV cache compression for the GPU plugin, cutting that overhead roughly in half compared to INT8 and by two-thirds compared to FP16. This post walks through what it is, how to enable it, and what you can expect in practice.

---

## What Is the KV Cache and Why Does Compression Matter?

Every transformer model maintains a **key-value (KV) cache** during generation. For each attention layer, the keys and values computed for all previously seen tokens are stored so they do not have to be recomputed on the next step. This is the core of efficient autoregressive inference—but it comes at a cost.

The KV cache size scales with:
- **context length** (prompt + tokens generated so far)
- **number of layers and attention heads**
- **precision** of the stored tensors

For a model like Llama-3-8B at 17k tokens and FP16 precision, the KV cache alone consumes over 2 GB of device memory. On a discrete GPU with limited VRAM or an integrated GPU sharing system DDR bandwidth, this is a hard constraint on what context lengths are practical.

**Compressing the KV cache from FP16 to INT8** cuts that in half. **Going further to INT4** brings it down to roughly one-third of the FP16 baseline. OpenVINO has supported INT8 KV cache as the default option, and in 2026.2, INT4 is now available.

### Default Behavior

OpenVINO GPU plugin applies **INT8 KV cache compression by default**. If you are already using OpenVINO for LLM inference on GPU, your KV cache is most likely already in INT8 unless you explicitly overrode the setting.

INT4 must be **manually enabled**.

---

## What INT4 KV Cache Compression Means

When INT4 KV cache is enabled, keys and values are quantized from FP16 down to 4-bit integers before being written into the memory. During the attention computation, they are dequantized on the fly.

Internally, `i4` and `u4` are treated identically—the plugin normalizes `i4` to `u4` at compile time. Likewise, `u8` is normalized to `i8`. You can use either spelling when setting the property; the behavior is the same.

The quantization scheme depends on the attention backend:

**Paged Attention backend** (recommended):
- **Keys** are quantized with per-channel scales (`BY_CHANNEL` mode, group size = 16).
- **Values** are quantized with per-token scales (`BY_TOKEN` mode).

This asymmetry is intentional. Channel-wise quantization of keys tends to preserve attention accuracy better, while token-wise quantization of values is more efficient for the decode-phase kernel.

**SDPA backend** (non-paged attention):
- Both **keys and values** are quantized with per-token scales (`BY_TOKEN` mode).

The lack of per-channel key quantization in the SDPA path means accuracy impact is more likely compared to the Paged Attention path.

---

## How to Enable INT4 KV Cache

Here's how to enable int4 kv cache in GenAI benchmark.py tool.

```bash
python tools/llm_bench/benchmark.py \
  -m /path/to/model \
  -d GPU \
  -lc '{"KV_CACHE_PRECISION": "u4"}'
```

### Reverting to a Higher Precision

If you need to disable KV cache compression entirely (for maximum accuracy or debugging):

```bash
python tools/llm_bench/benchmark.py \
  -m /path/to/model \
  -d GPU \
  -lc '{"KV_CACHE_PRECISION": "f16"}'
```

---

## Memory Usage Impact

The numbers below were measured with **Llama-3-8B-Instruct** on an **Intel Arc B580** (discrete GPU, Linux). KV cache sizes are consistent among different model weight precision.

*DISCLAIMER: The impact may vary depending on the system, model and usage.*

### 8k-token prompt

| KV Cache Precision | KV Cache Size | vs FP16 |
|---|---|---|
| FP16 (uncompressed) | 1024.12 MB | baseline |
| INT8 (default in OV) | 584.12 MB | −43% |
| **INT4** | **328.12 MB** | **−68%** |

### 17k-token prompt

| KV Cache Precision | KV Cache Size | vs FP16 |
|---|---|---|
| FP16 (uncompressed) | 2254.27 MB | baseline |
| INT8 (default in OV) | 1285.76 MB | −43% |
| **INT4** | **722.26 MB** | **−68%** |

The savings in **INT4** are consistent across context lengths at approximately **−44% vs INT8** and **−68% vs FP16**. The savings are not exactly 50% relative to INT8 because group-wise quantization stores per-group scale and zero-point alongside the compressed values.

The practical implication: a context length that exhausts available DDR at INT8 precision may fit at INT4.

---

## Performance Impact

Memory savings translate into performance on IO-bound case because less IO is required to generate a token.

The following results were measured with **Llama-3.1-8B-Instruct** on an **Intel Arc B390 integrated GPU** (with 9600 MT/s DDR), measuring decode latency per output token.(Prefill is excluded)

*DISCLAIMER: The impact may vary depending on the system, model and usage.*

### 16k-token prompt

| KV Cache Precision | Decode Latency | Speedup vs FP16 |
|---|---|---|
| FP16 | 68.29 ms/token | baseline |
| INT8  (default in OV) | 60.69 ms/token | 1.13× |
| **INT4** | **57.86 ms/token** | **1.18×** |

### 34k-token prompt

| KV Cache Precision | Decode Latency | Speedup vs FP16 |
|---|---|---|
| FP16 | 102.67 ms/token | baseline |
| INT8 (default in OV) | 86.14 ms/token | 1.19× |
| **INT4** | **80.48 ms/token** | **1.28×** |

---

## Accuracy Impact

Quantizing the KV cache introduces more quantization error into the attention computation. Based on internal validation:

- **INT4-weight models** (e.g., models compressed with NNCF 4-bit weight quantization): KV cache INT4 compression shows **equivalent accuracy** compared to INT8 KV cache. The model has already been optimized to tolerate quantization noise, and the KV cache error stays within that tolerance.

- **INT8-weight (or FP16-weight) models**: INT4 KV cache may show accuracy deviation. The degree depends on the model and task.

**It is recommended to validate accuracy on your target task before deploying INT4 KV cache in production.** A simple sanity check is to compare model outputs on a representative sample of prompts between INT8 and INT4 KV cache settings.

---

## Known Limitations

- **Cache rotation is not supported.** Serving configurations that combine KV cache block eviction with RoPE positional correction (cache rotation) are incompatible with INT4 KV cache. This does not affect typical single-session inference or standard prefix caching.
- **GPU plugin only.** INT4 KV cache compression is not available on the CPU plugin.
- **By-channel key quantization requires Paged Attention.** The `BY_CHANNEL` quantization mode for keys (which offers better accuracy) is only available when using the Paged Attention backend. When running with the SDPA backend (non-paged attention), keys fall back to `BY_TOKEN` quantization, which is more likely to introduce accuracy deviation. **Paged Attention is the recommended inference flow** and should be preferred when using INT4 KV cache compression.

---

## Summary

| | FP16 KV Cache | INT8 KV Cache (default) | INT4 KV Cache (new) |
|---|---|---|---|
| Memory (8k ctx) | 1024 MB | 584 MB | 328 MB |
| Memory (17k ctx) | 2254 MB | 1286 MB | 722 MB |
| Decode latency (iGPU, 34k) | 102.67 ms/tok | 86.14 ms/tok | 80.48 ms/tok |
| Accuracy (4-bit weight model) | best | best | equivalent to int8 |
| Accuracy (8-bit weight model) | best | best | may see deviation |
| Requires explicit opt-in | yes | no (default) | **yes** |

INT4 KV cache compression is impactful when you are memory-constrained and want to serve longer contexts.

Enable it with `"KV_CACHE_PRECISION": "u4"` and validate performance and accuracy on your workload. For best accuracy results, use the **Paged Attention backend**, which enables by-channel key quantization. If you are on the SDPA backend, be aware that per-token quantization is used for keys as well, and accuracy impact is more likely—validate carefully before deploying.
