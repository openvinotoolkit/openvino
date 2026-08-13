# GGUF Frontend — Supported Models

This document lists the model architectures the GGUF frontend can convert and run
end-to-end. An architecture is listed as **Supported** only when at least one *real*
(non-synthetic) model of that architecture has been verified to load, convert, and
produce coherent output through the frontend.

Verification is done by running a real `.gguf` through the OpenVINO backend swap in
`llama.cpp` (`llama-completion`, CPU device, stateful execution) and confirming the
generated text is coherent and matches the pure-ggml CPU reference for the same prompt.

## Supported architectures

Each row was verified with the named real model.

| Architecture | Verified model | Notes |
|---|---|---|
| `llama`   | TinyLlama-1.1B-Chat v1.0 (Q4_K_M) | Dense; standard RoPE + GQA. |
| `qwen2`   | Qwen2.5-0.5B-Instruct (Q8_0)      | Dense. |
| `qwen3`   | Qwen3-0.6B (Q8_0)                 | Dense; QK-norm. |
| `qwen3moe`| Qwen3-0.9B-A0.6B (Q4_K_M), Qwen3-4B (Q4_K_M) | Mixture-of-experts (`mul_mat_id`). |
| `olmoe`   | OLMoE-1B-7B-0924-Instruct (Q4_0)  | Mixture-of-experts. |
| `gemma3`  | gemma-3 family                    | Mixed sliding-window / global RoPE. |
| `gemma4`  | gemma-4-E4B-it (Q4_K_M)           | Per-op RoPE (SWA vs global); f16 KV cache. |
| `qwen35`  | Qwen3.5-4B (Q4_K_M)               | Hybrid GatedDeltaNet + full-attention layers; partial-rotary IMROPE; interleaved Q/gate joint projection; f16 KV cache. |

`llama`, `qwen2`, `qwen3`, `olmoe`, `gemma4`, and `qwen35` were verified with a fresh
end-to-end run of the model named above. `qwen3moe` and `gemma3` were verified in earlier
development on the models named above.

Quantization formats verified in the above runs: `Q2_K`, `Q4_0`, `Q4_K_M`, `Q6_K`,
`Q8_0`. The frontend weight path also handles `Q4_1`, `Q5_0`, `Q5_1` and the F16/F32
paths; these are exercised by the unit tests but have not each been tied to a specific
end-to-end real-model run.

## How verification was performed

```sh
GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=1 \
  llama-completion -m <model>.gguf -p "The capital of France is" -n 12 -no-cnv --no-warmup
```

A run counts as verification only when the output is coherent (e.g. completes
"...is Paris") and consistent with the pure-ggml CPU backend on the same prompt. A model
that loads but emits garbage (e.g. `hunyuan`) is **not** counted as supported.

## Architectures accepted by the native `.gguf` builder

Everything above is about the **llama.cpp cgraph** path. This section covers the *other*
decoder — the native `.gguf` builder (`TransformerBuilder` in
[`src/builder/gguf_builder.cpp`](../src/builder/gguf_builder.cpp)), which is what
OpenVINO GenAI uses. The two paths share all op
translators but have separate architecture lists.

The builder's accept-list is the union of two sets, both defined at the bottom of
`gguf_builder.cpp`:

- **`verified_archs()`** — convert + compile + generation checked against a reference on a
  real checkpoint.
- **`experimental_archs()`** — expected to work via the builder's GGUF-tensor-table
  auto-detection, but not end-to-end verified. These convert and emit a one-time
  `OPENVINO_WARN` so callers know they are best-effort.

Anything not in either set is rejected with an explicit `OPENVINO_ASSERT` at load time
rather than converting into a silently wrong graph.

### `verified_archs()` — 14 architectures

| Architecture | Notes |
|---|---|
| `llama` | llama-2 / llama-3 |
| `qwen2` | qwen2 / qwen2.5 |
| `qwen3` | QK-norm |
| `phi3` | fused QKV |
| `minicpm` | NORMAL rope + scalar embedding/residual/logit scales |
| `hunyuan-dense` | |
| `olmoe` | OLMoE 1B-7B (MoE) |
| `qwen3moe` | Qwen3 MoE; same topology as `olmoe` |
| `qwen35` | Qwen3.5/3.6 (and the Ternary-Bonsai backbone): hybrid Gated-DeltaNet + full attention, M-RoPE, interleaved query+gate projection. Greedy / batch 1 only |
| `gpt-oss` | MoE + attention sinks + SWA + OAI gated activation |
| `gemma` | Gemma 2B / 7B |
| `gemma2` | post-norms + attention soft-cap |
| `gemma3` | post-norms + final logit soft-cap |
| `gemma4` | SWA, per-layer embeddings, shared KV |

### `experimental_archs()` — 16 architectures

| Architecture | Notes |
|---|---|
| `llama-embed` | Bidirectional LLaMA (embedding model, no causal mask) |
| `exaone4` | EXAONE 4.0: NEOX rope, post-norms (attn + ffn) |
| `plamo3` | PLaMo-3: NEOX rope, post-norms (attn + ffn) |
| `smollm3` | SmolLM3: NORMAL rope + SWA |
| `hunyuan-moe` | NEOX rope, MoE routing, QK-norm |
| `glm4moe` | GLM 4.5 MoE: 1 dense lead layer, MoE + attn post-norm |
| `exaone-moe` | EXAONE MoE: SWA + MoE, shared expert |
| `minimax-m2` | Minimax M2: pure MoE |
| `ernie4_5-moe` | Ernie 4.5 MoE: NORMAL rope, dense lead layers + MoE stride |
| `bailingmoe2` | BailingMoe V2: MoE + shared expert + QK-norm |
| `maincoder` | Maincoder-1B: NORMAL rope, QK-norm (auto-detected) |
| `mistral3` | Ministral-3B: NORMAL rope, dense |
| `muse-glimmer` | Muse Glimmer (Meta Onyx): NORMAL rope on SWA layers only (global layers are NoPE), sigmoid attention output gate, QK-norm, pre+post norms, final logit soft-cap |
| `mellum` | JetBrains Mellum: pure MoE |
| `deepseek2-ocr` | DeepSeekOCR: dense lead layers + MoE |
| `jais2` | JAIS-2: dense (biases auto-detected) |

RoPE flavor is **not** in these tables because it is a separate switch: archs listed in
`arch_uses_neox_rope()` use NEOX (rotate-halves), everything else uses NORMAL (rotate
consecutive pairs). Adding an arch to the accept-list without also classifying its RoPE is
the most common way to get a model that loads and produces garbage.

### Measured status through OpenVINO GenAI

Every architecture above was run through GenAI on CPU (`gguf_arch_check`, greedy, SDPA
backend) on the checkpoint named below. "Generates" means the model answered *"The capital
of France is"* correctly and coherently; **`llama.cpp` ref** is the same `.gguf` through
`llama-cli` on the default ggml CPU backend, which distinguishes a frontend bug from a
model/checkpoint that is simply weak on the prompt.

| Arch | Set | Model used | GenAI | llama.cpp ref |
|---|---|---|---|---|
| `llama` | verified | Llama-3.2-1B-Instruct Q4_K_M | generates | generates |
| `qwen2` | verified | Qwen2.5-0.5B-Instruct Q4_K_M | generates | generates |
| `qwen3` | verified | Qwen3-0.6B Q8_0 | generates (reasoning preamble) | same |
| `phi3` | verified | Phi-3-mini-4k-instruct Q4 | generates | generates |
| `minicpm` | verified | MiniCPM-2B-dpo Q4_K_M | generates | generates |
| `hunyuan-dense` | verified | Hunyuan-0.5B-Instruct Q4_K_M | **degenerate** | generates |
| `olmoe` | verified | OLMoE-1B-7B-Instruct Q4_K_M | generates | generates |
| `qwen3moe` | verified | Qwen3-0.9B-A0.6B Q4_K_M | **degenerate** | generates |
| `gpt-oss` | verified | gpt-oss-20b MXFP4 | generates (harmony format) | same |
| `gemma` | verified | gemma-2b Q4_K_M | **throws** (SDPA shape mismatch) | degenerate too |
| `gemma2` | verified | gemma-2-2b-it Q4_K_M | **degenerate** | generates |
| `gemma3` | verified | gemma-3-1b-it Q4_K_M | generates | generates |
| `gemma4` | verified | gemma-4-E4B-it Q4_K_M | generates | generates |
| `llama-embed` | experimental | llama-nemotron-embed-1b-v2 Q4_K_M | repeats (embedding model) | degenerate too |
| `exaone4` | experimental | EXAONE-4.0-1.2B Q4_K_M | **degenerate** | generates |
| `plamo3` | experimental | plamo-3-nict-2b-base Q4_K_M | **degenerate** | degenerate too |
| `smollm3` | experimental | SmolLM3-3B Q4_K_M | generates (reasoning preamble) | same |
| `maincoder` | experimental | Maincoder-1B Q4_K_M | generates | generates |
| `mistral3` | experimental | Ministral-3-3B-Instruct-2512 Q4_K_M | generates | generates |
| `muse-glimmer` | experimental | Muse-Glimmer-30B Q4_0 | generates | generates |
| `qwen35` | verified | Qwen3.5-0.8B Q8_0 | generates | generates |
| `qwen35` (Bonsai) | verified | Ternary-Bonsai-27B Q2_g64 | generates | generates |
| `deepseek2-ocr` | experimental | deepseek-ocr-2 Q4_K_M | generates | generates |
| `ernie4_5-moe` | experimental | ERNIE-4.5-21B-A3B Q4_K_M | **degenerate** (blank) | generates |
| `bailingmoe2` | experimental | Ling-mini-2.0 Q2_K | generates | generates |
| `mellum` | experimental | Mellum2-12B-A2.5B-Instruct Q4_K_M | generates | generates |
| `hunyuan-moe` | experimental | — | not tested (no checkpoint) | — |
| `glm4moe` | experimental | — | not tested (smallest GLM-4.5-Air ≈ 40 GiB) | — |
| `exaone-moe` | experimental | — | not tested (smallest ≈ 9 GiB, 32B) | — |
| `minimax-m2` | experimental | — | not tested (smallest ≈ 78 GiB) | — |
| `jais2` | experimental | — | not tested (no checkpoint) | — |

Two caveats on reading this table. `llama-embed` is an *embedding* model, so degenerate
greedy completion is expected of it, not a defect. `gemma` (v1 base) and `plamo3` (base, not
instruct) are degenerate on the reference too, so those rows are checkpoint/prompt artifacts
rather than frontend bugs.

That leaves **5 architectures that generate correctly under llama.cpp but not through the
builder** — `hunyuan-dense`, `qwen3moe`, `gemma2`, `exaone4` and `ernie4_5-moe` (blank output) —
i.e. real conversion defects, plus `gemma`, which throws instead of converting cleanly. Four of
those (`hunyuan-dense`, `qwen3moe`, `gemma2`, `gemma`) are in `verified_archs()`, so that set is
currently **optimistic** and should be re-validated before it is relied on.

`qwen35` was validated the same way muse-glimmer was, feeding llama.cpp's own token ids and
comparing greedy output. On Qwen3.5-0.8B-Q8_0 and on Ternary-Bonsai-27B-Q2_g64 the frontend
reproduces llama.cpp **token for token** (`" Paris.\nThe capital of France is Paris. ..."` and
`" Paris. Paris is the largest city in France. Paris is the most popular"` respectively). Final
logits agree to 1.0% on the 0.8B (sum -771776 vs -779763) and 0.12% on Bonsai (-812702 vs
-813701); the 0.8B figure is in line with the *verified* `qwen3` arch measured through the same
harness, so it is dequant/driver noise rather than an arch defect.

**`qwen35` is greedy / batch-1 only.** The recurrent conv and delta states are a single
static-shaped block with no batch axis, and `MakeStateful` does not reorder them by `beam_idx`
the way it reorders a KV cache. Beam search or batch > 1 therefore **fails at inference** with a
shape mismatch on the conv window's `Concat` -- it does not silently mix state across beams, so
no wrong output can be produced. Prefix caching and PagedAttention are unavailable for the same
reason: a recurrent state cannot be re-derived from a cached prefix, and cannot be paged.

What "verified" rests on for this arch, since a hybrid stack is easy to get subtly wrong:

* Token-exact greedy agreement with llama.cpp on **two real checkpoints** of different size and
  quantization (Qwen3.5-0.8B Q8_0, Ternary-Bonsai-27B Q2_g64), through GenAI's own harness.
* A **five-prompt** raw-completion sweep on the 0.8B. Three match llama.cpp exactly; two diverge
  mid-continuation. Both divergences are near-ties, and the *already verified* `qwen3` arch
  diverges on the same prompts in the same way ("20 years old" vs "22 years old"), so this is the
  frontend's numerical baseline rather than a qwen35 defect.
* A **prefill-vs-decode consistency** check (generate N tokens, then re-prefill each own-output
  prefix and compare the next token). `qwen35` scores 1 mismatch in 16 -- identical to `qwen3`
  (1/16) at the identical position. `llama` scores 0/16. The GDN recurrence is computed chunk-wise
  during prefill and step-wise during decode, which are mathematically equal but not bit-equal, so
  a near-tie can flip; llama.cpp has the same two code paths.
* Final logits within 1.0% (0.8B) and 0.12% (Bonsai) of llama.cpp on the sum over the vocabulary.

A caveat on the Bonsai artifacts: **`Ternary-Bonsai-27B-Q2_0.gguf` is not upstream `Q2_0`.**
It does not load in llama.cpp either (`tensor 'output_norm.weight' has offset 337715200,
expected 357580800`; the ratio 0.94444 is exactly `(34/128)/(18/64)`). The file is packed
**g128** -- one f16 scale per 128 weights, 2.125 bits/weight -- while `GGML_TYPE_Q2_0` is
**g64**, 18 bytes per 64 weights, 2.25 bits/weight. The model card says as much, calling the
deployed format "Q2_0_g128" and publishing a separate group-64 pack "matching the 64-value-group
Q2_0 packing in llama.cpp". Use `Ternary-Bonsai-27B-Q2_g64.gguf`; the g128 pack needs PrismML's
fork. The frontend rejects it safely (`tensor 'blk.63.ffn_up.weight' data runs past EOF`) rather
than dequantizing garbage.

`muse-glimmer` needs a footnote of its own, because it is the one arch whose row was decided
by the *tokenizer*, not the graph. Fed llama.cpp's own token ids, the converted graph
reproduces llama.cpp token-for-token: for `<|begin_of_text|>The capital of France is` all 32
greedy tokens are identical, and the final logits agree to within the frontend's ordinary
dequantization noise (sum -821515 vs -824015, 0.3%, *tighter* than the `qwen3` Q4_0 control
at 0.17% on values ~3x smaller). Through GenAI it initially looked degenerate, because
GenAI's GGUF tokenizer honored `tokenizer.ggml.add_bos_token` only on the SentencePiece
(`tokenizer.ggml.model = llama`) path; on the BPE (`gpt2`) path it built no CombineSegments
node, so the leading BOS was silently dropped. Muse Glimmer is `gpt2` + `add_bos_token = true`
and is BOS-sensitive, so it looped on `The capital of France is`. Without the BOS the
converted graph picks `" The"` at that position (top-8 `589=12.42 5422=10.97 1573=10.69`);
with it, `" It"` (`1573=17.34`), which is what llama.cpp emits. The gap was arch-independent
(`llama3` and `mistral3` lost their BOS the same way) and lived in GenAI, not in this
frontend; it is fixed in `gguf_tokenizer.cpp` by emitting BOS/EOS as a CombineSegments
segment on every tokenizer path.

### Measured performance and memory (OpenVINO GenAI, CPU)

Same runs as the table above. i9-12900K (16C/24T), OV defaults. Prefill = prompt tokens /
TTFT on a ~90-340-token prompt; decode = 1/TPOT over 32 greedy tokens, steady-state
iteration. `peak RSS` and `peak anon` are the maxima of `Rss:`/`Anonymous:` from
`/proc/self/smaps_rollup`, sampled every 20 ms in-process; `anon` is the part that
genuinely requires RAM (see [`frontend_design.md`](frontend_design.md) on the memory model).
`load` is `.gguf` → OV graph → `compile_model`.

| Arch | Model MiB | load s | prefill t/s | decode t/s | peak RSS MiB | peak anon MiB |
|---|---|---|---|---|---|---|
| `qwen2` | 468 | 4.7 | 1196.3 | 78.68 | 1492 | 1418 |
| `qwen3` | 609 | 4.9 | 865.1 | 64.94 | 1612 | 1538 |
| `hunyuan-dense` | 338 | 4.2 | 630.3 | 62.93 | 1545 | 1473 |
| `gemma3` | 768 | 1.9 | 662.6 | 43.76 | 2610 | 2536 |
| `qwen3moe` | 531 | 5.7 | 158.0 | 44.96 | 1987 | 1912 |
| `maincoder` | 640 | 5.1 | 324.3 | 36.28 | 2321 | 2245 |
| `llama-embed` | 770 | 6.0 | 266.9 | 35.75 | 2601 | 2529 |
| `exaone4` | 774 | 4.3 | 236.5 | 35.54 | 2594 | 2520 |
| `olmoe` | 4018 | 12.9 | 86.6 | 35.17 | 11760 | 11684 |
| `llama` | 770 | 6.0 | 323.8 | 35.08 | 2628 | 2556 |
| `bailingmoe2` | 5573 | 44.9 | 108.1 | 26.97 | 36956 | 36237 |
| `deepseek2-ocr` | 1859 | 7.4 | 299.2 | 71.38 | 5392 | 5318 |
| `mellum` | 7697 | 27.7 | 70.8 | 21.48 | 21282 | 21194 |
| `gemma` | 1425 | 3.8 | 149.6 | 18.47 | 4570 | 4501 |
| `plamo3` | 1574 | 3.3 | 154.9 | 16.87 | 5661 | 5594 |
| `smollm3` | 1826 | 8.4 | 121.9 | 14.88 | 5972 | 5895 |
| `minicpm` | 1649 | 3.3 | 126.0 | 14.82 | 5041 | 4968 |
| `ernie4_5-moe` | 12873 | 46.6 | 45.8 | 14.67 | 36551 | 36460 |
| `gemma2` | 1629 | 3.4 | 132.8 | 14.35 | 5692 | 5617 |
| `mistral3` | 2047 | 8.8 | 103.6 | 12.96 | 6871 | 6796 |
| `phi3` | 2282 | 4.4 | 108.3 | 11.84 | 8197 | 8130 |
| `gemma4` | 4746 | 9.6 | 63.6 | 9.24 | 12309 | 11953 |
| `gpt-oss` | 11548 | 89.3 | 18.7 | 5.48 | 123720 | 123493 |
| `qwen35` (Qwen3.5-0.8B Q8_0) | 795 | 1.5 | 537.4 | 41.38 | 2292 | 2207 |
| `qwen35` (Bonsai-27B Q2_g64) | 7234 | 28.7 | 21.4 | 3.75 | 24009 | 23903 |
| `muse-glimmer` | 15512 | 28.3 | 28.9 | 2.66 | 37457 | 37357 |

Numbers from architectures marked degenerate above still describe real compute cost (the
graph runs, it is just numerically wrong), so they are kept for completeness.

The two `qwen35` rows come from the same `gguf_arch_check` harness as every other row: GenAI
runs this architecture now that `MakeStateful` also rewrites the recurrent conv/delta states and
`AdaptToGenAI` expands `position_ids` into M-RoPE's four sections. Both reproduce llama.cpp
token-for-token; see the numerical notes below.

Against llama.cpp on the same host, `qwen35` decodes at **0.74x** (41.4 vs 56.1 tok/s) -- the
usual SDPA-path ratio -- while prefill is not directly comparable here because the two harnesses
use different prompt lengths (95 vs 5 tokens); on the matched 5-token prompt the frontend
prefills 1.10x faster (191.4 vs 174.5 tok/s). Bonsai inverts the decode picture dramatically:
**6.5x faster** (3.75 vs 0.58 tok/s). That is not an OpenVINO win so much
as an upstream gap -- ggml ships no x86 SIMD kernel for `Q2_0`, so `ggml_vec_dot_q2_0_q8_0`
falls back to the generic scalar reference, while the frontend lowers Q2_0 into the ordinary
u2 compressed-weights MatMul the CPU plugin already optimizes. (PrismML's own fork ships
tuned CUDA/Metal kernels; this comparison is upstream-CPU vs OpenVINO-CPU.) Memory is the
other side of that trade: llama.cpp mmaps the weights and peaks at 7.5 GiB for Bonsai, the
frontend materializes decompression constants and peaks at 22.7 GiB (3.1x the file).

`muse-glimmer` is the largest checkpoint in the table (30B, 15.5 GiB) and is memory-bandwidth
bound at 2.66 tok/s decode; llama.cpp on the same file and host does 3.46 tok/s, so the
frontend lands at **0.77x llama.cpp**, in line with the 0.51-0.64x SDPA ratios measured on
the smaller models below. Peak anon is 2.4x the file, better than the 3-4x typical elsewhere,
because Q4_0 stays 4-bit and only the Q6_K tensors are requantized to Q8_0_C.

One outlier remains: `gpt-oss` peaks at **124 GiB from an 11.5 GiB file (11x)**, versus a
typical 3-4x elsewhere. On a smaller-RAM host it would OOM. The cause is the compressed-weights
type gate on the MoE expert matmul described below — for gpt-oss the expert type is MXFP4
(`f4e2m1`), which the frontend dequantizes on-graph in `MUL_MAT_ID` rather than routing through
`GatherMatmul` at all, so the plugin-side widening does not reach it.

### Measuring performance correctly

Three traps have each produced a wrong published number at least once. Read this before
benchmarking, especially when comparing against llama.cpp.

**1. Disable prefix caching when measuring prefill under PagedAttention.** `ATTENTION_BACKEND=PA`
routes through GenAI's ContinuousBatching adapter, and `get_latency_oriented_scheduler_config()`
(GenAI `src/cpp/src/utils.cpp`) sets `enable_prefix_caching = true` by default. Benchmarks
typically repeat one fixed prompt for N iterations to amortize the first-request dynamic-shape
compile — with prefix caching on, **every iteration after the first is a cache hit**, so the
reported TTFT is not prefill work at all. On Llama-3.2-1B this reads 125 ms cached vs 300 ms
uncached (SDPA measures 304 ms): the cache made PA look 2.4x faster at prefill than an identical
computation. Pass an explicit scheduler config with it off:

```cpp
ov::genai::SchedulerConfig sched;
sched.max_num_batched_tokens = std::numeric_limits<std::size_t>::max();  // as the latency default
sched.enable_prefix_caching = false;
props[ov::genai::scheduler_config.name()] = sched;
```

Note the asymmetry that makes this specifically a *comparison* hazard: SDPA ignores this knob
entirely, and neither llama.cpp reference path caches across runs. `llama-bench` calls
`llama_memory_clear()` inside the rep loop before the timer starts (its state-reuse path is gated on
`-d/--n-depth > 0`, which defaults to 0); `llama-cli` is single-shot and its `--prompt-cache`
defaults to empty. So a cached PA number is being compared against two uncached ones. Sanity check:
run llama-bench with `-r 5` and confirm variance stays under ~1% — a cache hit shows as a large drop
after rep 0, not as noise. Prefix caching is a real PA capability worth reporting *separately*; it
just is not prefill throughput.

**2. Confirm PA is actually in use — the fallback is silent.** GenAI catches a PA initialization
failure and falls back to SDPA with only a `GENAI_WARN` (`src/cpp/src/llm/pipeline.cpp`), and the
default log level is `ERR` (`src/cpp/src/logger.cpp`), so **the warning is invisible unless you set
`OPENVINO_LOG_LEVEL=4`**. Correct output and plausible timings therefore prove nothing. Counting
`PagedAttention` in the `ov::Model` is also insufficient — that is the graph handed *to* the plugin.
Check the compiled **runtime** graph:

```cpp
auto rt = compiled_model.get_runtime_model();
for (const auto& op : rt->get_ops())
    hist[op->get_rt_info().at("layerType").as<std::string>()]++;
```

For Llama-3.2-1B (16 layers) the two backends must look like this — note `MemoryInput`/`MemoryOutput`
disappearing, since PA replaces the stateful KV cache with the plugin's block-table cache. A rename
alone would not do that:

| runtime `layerType` | SDPA | PA |
|---|---|---|
| `PagedAttention` | 0 | 16 |
| `ScaledDotProductAttention` | 16 | 0 |
| `MemoryInput` / `MemoryOutput` | 32 / 32 | 0 / 0 |

**3. Drop iteration 0 and pin the comparison.** Iteration 0 carries the first-request dynamic-shape
compile (several hundred ms to seconds); average iterations 1..N-1 for steady state. Compare on the
same `.gguf` file, the same prompt text, and the same `n_ctx` — llama.cpp preallocates the whole
`n_ctx` KV cache up front while OV's stateful cache grows on demand, so a mismatched context length
makes the memory figures incomparable. Also record the thread counts: llama.cpp auto-selects
P-cores only (8 on an i9-12900K) where OV uses all 24 by default, which is not a like-for-like
core budget unless equalized.

Putting it together — the three commands behind the table below. llama.cpp is measured twice because
`llama-bench` gives steady-state kernel throughput with no process/load overhead, while `llama-cli`
walks the same end-to-end path as the GenAI sample and so is the fair peak-RSS comparison:

```sh
# steady-state kernel throughput (cache cleared per rep; -r 5 to confirm low variance)
llama-bench -m "$MODEL" -p 128 -n 128 -r 5

# end-to-end, for max-RSS parity with the GenAI sample
/usr/bin/time -v llama-cli -m "$MODEL" -p "$PROMPT" -n 128 -c 1024 \
    -no-cnv -st --temp 0 --seed 1 --no-warmup --ignore-eos

# GenAI, once per backend; the sample turns prefix caching off for PA (trap 1) and
# reports per-iteration TTFT/TPOT so iteration 0 can be dropped (trap 3)
/usr/bin/time -v bench_gguf_perf "$MODEL" "$PROMPT" 128 4 {SDPA|PA}
```

`bench_gguf_perf` is the GenAI sample at `samples/cpp/text_generation/bench_gguf_perf.cpp`; keep the
same `-c/n_ctx` on both sides and the same prompt text everywhere.

#### PagedAttention vs SDPA vs llama.cpp (measured under the rules above)

i9-12900K, Q4_K_M, 128 generated tokens, 4 iterations with iteration 0 dropped, `n_ctx=1024`,
prefix caching **off**, PA presence confirmed in the runtime graph for every row. llama.cpp is its
default ggml CPU backend (`llama-bench pp128/tg128`), 8 threads by its own auto-selection.

| Model | prompt tok | prefill t/s (lcpp / SDPA / PA) | decode t/s (lcpp / SDPA / PA) | PA/SDPA | PA/lcpp | peak RSS GB (lcpp / SDPA / PA) |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | 87 | 528 / 291 / 291 | 73.7 / 39.7 / 41.3 | 1.04 | 0.56 | 1.30 / 2.51 / 2.51 |
| Maincoder-1B | 68 | 591 / 405 / 409 | 84.1 / 43.9 / 45.7 | 1.04 | 0.54 | 1.10 / 2.24 / 2.24 |
| gemma-3-1b | 75 | 411 / 467 / 429 | 75.5 / 47.1 / 47.1 | 1.00 | 0.62 | 0.92 / 2.49 / 2.49 |
| Ministral-3-3B | 631 | 158 / 111 / 120 | 27.1 / 14.8 / 15.0 | 1.01 | 0.55 | 3.60 / 6.67 / 6.65 |
| SmolLM3-3B | 302 | 173 / 135 / 136 | 30.2 / 16.8 / 17.2 | 1.02 | 0.57 | 3.21 / 5.71 / 5.70 |
| mistral-7b-v0.1 | 55 | 68 / 52 / 56 | 13.7 / 6.94 / 6.99 | 1.01 | 0.51 | 7.37 / 11.81 / 11.82 |
| Ministral-8B | 53 | 69 / 40 / 39 | 12.8 / 7.01 / 7.05 | 1.01 | 0.55 | 7.92 / 13.10 / 13.11 |
| gemma-4-E4B | 58 | 111 / 55 / 56 | 18.2 / 11.1 / 11.6 | 1.04 | 0.64 | 6.96 / 12.36 / 11.92 |

**PA vs SDPA: parity.** Decode 1.00-1.04x (PA marginally ahead on all 8), prefill within +-8%, peak
RSS within 0.2% except gemma-4 where PA is 0.44 GB lower. Enabling PA costs nothing; the reason to
use it is that continuous batching, prefix caching and multi-sequence serving become available at
all, which the SDPA-only graph could not do.

**PA vs llama.cpp: decode 0.51-0.64x**, prefill 0.55-1.04x, peak RSS 1.7-2.4x. These ratios match
what the SDPA path already measured, so PA neither introduces nor closes that gap — see
[`frontend_design.md`](frontend_design.md) on the memory model for the RSS side.

### MoE expert weights and the compressed-weights type gate

Worth knowing when picking a quantization for a MoE model, though the handling is entirely
plugin-side. MoE expert weights do not go through `FullyConnected`: `MUL_MAT_ID` lowers to the
CPU plugin's `GatherMatmul` (equally, to `GroupedMatMul` on the public-op side — on CPU
`ConvertGroupedMatMulToGatherMatmul` rewrites it into the same node *before* the compression
pass, so the two are indistinguishable here). That node accepts a **narrower set of compressed
weight types than `FullyConnected` does**:

| | accepted compressed weight types |
|---|---|
| `FullyConnected` | `u8, i8, u4, i4, nf4, f4e2m1, u2` |
| `GatherMatmul` / GPU grouped-matmul | `u8, i8, u4, i4` |

If an expert weight's element type is outside the second set, `ConvertGatherMatmulToGather
MatmulCompressed` does not fire, the `Convert -> Subtract -> Multiply` dequantization block stays
in the graph, and constant folding materializes the experts **in f32** — a 16x expansion off a
2-bit type, i.e. far more than the quantization was saving.

Q2_K is the case this affects: its weights map to `u2`. The CPU plugin's
`WidenGatherMatmulWeights` pass handles it by re-emitting *expert* weight constants as `u4`
(lossless — raw Q2_K values are `[0..3]`, which fit a nibble) at 2x the weight bytes, which is
much cheaper than falling off the compressed path. Dense `u2` weights are left alone. This is a
plugin-side workaround for a missing `u2` expert-matmul executor and needs nothing from the
frontend, which emits plain `u2` either way. Measured on Q2_K models, peak anonymous memory:

| Model | file MiB | before | after |
|---|---|---|---|
| Qwen3-0.9B-A0.6B (`qwen3moe`) | 373 | 4251 | 2071 |
| Ling-mini-2.0 (`bailingmoe2`) | 5573 | 117245 | 36237 |

Decode also improves (bailingmoe2: 12.7 → 27.0 t/s) because the experts are no longer read from
f32.

## Adding a new architecture

Support for a new architecture is a combination of:
1. **Ops** — every ggml op in the model's compute graph must have a frontend translator
   (`src/op/<name>.cpp`) and backend admission.
2. **Weights** — every quantization format used by the model's tensors must be handled by
   the weight path (`src/quant/weights.cpp`).
3. **Real-model verification** — run a real `.gguf` end-to-end as above before adding the
   architecture to the Supported table.

For the native builder specifically, see
[`adding_an_architecture.md`](adding_an_architecture.md) — for a same-family arch the change
is usually just adding the name to `experimental_archs()` plus the `arch_uses_neox_rope()`
classification, and promotion to `verified_archs()` should require the GenAI-vs-llama.cpp
comparison above.
