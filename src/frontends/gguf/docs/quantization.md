# GGUF weights and execution precision

File quantization, frontend storage, inference arithmetic, and KV-cache precision are separate
choices. Support for a weight format does not establish model accuracy or device support.
The native format mapping lives in [`weights.cpp`](../src/quant/weights.cpp); parsing and tensor
extraction live in [`gguf.cpp`](../src/quant/gguf.cpp).

See [weight unpacking and its tradeoffs](weight_unpacking.md) for each format's block layout,
decompression algorithm, storage overhead and numerical or execution costs.

## Native weight formats

These are the stored Constant types after native frontend conversion. Weight nodes expose
F32 values to graph consumers; device compilation determines the execution representation.

| GGUF weight type | OpenVINO weight/code type | Scale type | Zero-point type | Representation |
|---|---|---|---|---|
| `F32`, `F16`, `BF16` | `f32`, `f16`, `bf16` respectively | — | — | Floating-point Constants |
| `Q1_0` | `u1` | `f16` | — | Binary codes, per-128 scale; `(2 * bit - 1) * scale` |
| `Q3_K`, `Q4_0` | `i4` | `f16` | — | Signed codes with grouped scales |
| `Q5_0`, `Q6_K`, `Q8_0` | `i8` | `f16` | — | Signed codes with grouped scales |
| `Q5_1` | `i8` | `f16` | `f16` | Five-bit codes widened to eight bits, fractional zero point |
| `Q8_K` | `i8` | `f32` | — | Signed codes with per-256 scales |
| `Q2_K` | `u2` | `f16` | `f16` | Two-bit codes with per-16 scales and fractional zero points |
| `Q2_0`, `PQ2_0`, `PTQ1_0` | `u2` | `f16` | `u8` | Per-64 or per-128 scales, zero point 1 |
| `Q4_1`, `Q4_K` | `u4` | `f16` | `u8` or `f16` | Ordinary matmul weights use integer-zero-point requantization by default |
| `Q5_K` | `u8` | `f16` | `u8` or `f16` | Ordinary matmul weights use eight-bit requantization with integer zero points |
| `MXFP4` | `f4e2m1` | `f8e8m0` | — | Four-bit float codes with per-32 scales |

Widening codes to a supported storage type is different from requantizing values. Native Q6_K
uses explicit F32 decompression arithmetic. Embedding/output tensors retain fractional zero
points where applicable. Q4_K/Q4_1 and Q5_K requantization add error beyond that already present
in the file; the frontend prints an accuracy notice once per process for each approximation.
Mixed file labels such as `Q4_K_M` describe a checkpoint's tensor mix, not one tensor type.

Prism `PQ2_0` (type 142) stores 128 ternary codes and an F16 scale in 34 bytes.
`PTQ1_0` (type 143) stores 128 dense trits and an F16 scale in 28 bytes. Both decode to u2
codes with integer zero point 1 and per-128 scales, without requantization.

Older Bonsai files used type 42 (`Q2_0`) for both per-64 and per-128 packing. The native parser
identifies the payload using tensor dimensions and aligned descriptor spans, and maps per-128
payloads to `PQ2_0` internally. The supplied-cgraph path must provide the actual quantization
type because raw bytes do not carry descriptor spans. Bonsai-2 also requires its
`prism.hadamard.*` metadata for activation transforms, including F16 checkpoints.

Q1_0 preserves binary storage in both input paths, including embedding/output tensors. Plugins
need compressed u1 matrix and lookup implementations to retain that storage during compilation;
the current CPU plugin expands these weights to floating point.

The supplied-cgraph/raw-byte weight path can have different representation choices, including
channel-wise Q8_0_C requantization for embedding/output and selected formats. Record which
path produced the model when comparing weights; native storage is not a claim about backend storage.

## Accuracy controls

For strict native CPU comparisons, use F32 inference, disable dynamic activation quantization,
and select the reference's KV-cache precision (the decoder fixture suite uses F16):

```python
compiled = core.compile_model(model, "CPU", {
    "INFERENCE_PRECISION_HINT": "f32",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0,
    "KV_CACHE_PRECISION": "f16",
})
```

Set `OV_GGUF_Q4_K_ZP_F16=1` **before starting the process** to preserve fractional Q4_K/Q4_1
zero points instead of their default u4 requantization. It is cached on first use; changing the
environment later in a notebook does not switch an already initialized process. The option
does not make every quantized format lossless. Fractional zero points can prevent compressed
FullyConnected fusion, slow matmul, or be unsupported by a device implementation.

llama.cpp's quantized kernels can quantize activations even when OpenVINO's dynamic activation
quantization is disabled. Report comparisons against quantized CPU arithmetic separately from
F32 arithmetic on the same represented weights. Improved greedy-token agreement can coexist
with worse full-logit error; use both metrics and the [acceptance criteria](testing.md#acceptance).

## Memory and packaging limits

Frontend compressed storage does not guarantee compressed execution. In particular, CPU compressed
expert matmul does not support u2 in this branch, so large Q2 MoE models can expand during
compilation. Include weight reorders and temporary buffers in memory estimates; a smaller context
does not remove those costs. Check [checkpoint-specific limits](supported_models.md#runtime-limitations).
