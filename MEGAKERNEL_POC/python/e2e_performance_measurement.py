"""End-to-end MegaKernel decode performance measurement.

Usage
-----
    source /opt/home/pwysocki/openvino_dist/setupvars.sh
    /opt/home/pwysocki/.venv/bin/python e2e_performance_measurement.py
    ... --device GPU.1 --frameworks decode_only optimum genai
    ... --tokens 200
    ... --only-framework decode_only
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DEFAULT_MODEL_DIR = HERE / "qwen3-0.6b-openvino-ir"
DEFAULT_DEVICE = "GPU.1"
BATCH = 1

DECODE_ONLY_PROMPT = "What is the capital of France?"

VERY_LONG_PROMPT = (
    "You are the performance lead for an engineering team deploying a compact "
    "decoder-only language model at the edge. Review the complete case study below "
    "and prepare a recommendation for the runtime and GPU compiler teams. Your answer "
    "must connect the observations rather than discuss each item in isolation.\n\n"
    "Business and workload context\n"
    "The product is an offline technical assistant used by field engineers in places "
    "where a network connection is unreliable. It summarizes equipment logs, answers "
    "questions about service manuals, and drafts short repair reports. Interactive "
    "requests arrive one at a time, so the normal serving batch is one. A typical user "
    "prompt contains 700 to 1800 tokens retrieved from local manuals, although diagnostic "
    "sessions can approach 4000 tokens. Most responses contain between 80 and 240 new "
    "tokens. Users notice the initial pause before text appears, but they are even more "
    "sensitive to uneven token delivery after streaming begins. The product requirement "
    "is a time to first token below 450 milliseconds and a sustained rate above 35 tokens "
    "per second without sending data to a server. Power consumption matters because the "
    "device may run from a battery for an entire shift.\n\n"
    "Hardware platform\n"
    "The target system contains a mobile processor with six CPU cores and an integrated "
    "GPU containing 64 execution units. CPU and GPU share 16 GB of LPDDR memory with a "
    "measured sustainable bandwidth of 72 GB per second under this workload. The GPU has "
    "a 4 MB last-level cache shared by all subslices, 128 KB of local memory per subslice, "
    "and a finite register file allocated among resident workgroups. The display engine "
    "refreshes two monitors from the same memory fabric. Firmware enforces a package power "
    "limit, so heavy CPU activity can reduce GPU frequency from 1.45 GHz to about 1.1 GHz. "
    "Long periods of full utilization eventually reach a thermal steady state that is "
    "lower than the short benchmark boost frequency. The operating system can preempt GPU "
    "work for display deadlines, but those interruptions are uncommon and usually short.\n\n"
    "Model and numerical format\n"
    "The model has 32 transformer layers, a hidden width of 2048, grouped-query attention, "
    "and roughly 600 million parameters. Weights are stored in a four-bit representation "
    "with group-wise scales and are unpacked near the matrix multiplication. Activations "
    "and the key-value cache use sixteen-bit floating point values. Rotary position "
    "embeddings are applied to queries and keys. Each layer performs input normalization, "
    "attention projections, cache update, attention score and value operations, an output "
    "projection, a residual addition, a second normalization, and a gated feed-forward "
    "network. At batch one, a decode iteration consumes one input token and produces one "
    "logit vector. The weights are much larger than the cache and cannot remain in the "
    "last-level cache between layers or between consecutive tokens.\n\n"
    "Prefill behavior\n"
    "During prefill, hundreds or thousands of prompt positions are available at once. "
    "Projection and feed-forward operations therefore become matrix-matrix multiplications "
    "with enough independent tiles to occupy the device. Their arithmetic intensity is "
    "higher because a weight tile can be reused across many tokens before eviction. The "
    "attention calculation grows with prompt length and also exposes parallel work across "
    "heads and query positions. Existing kernels already use large tiles and overlap some "
    "memory traffic with arithmetic. Profiling a 1024-token prompt attributes most prefill "
    "time to compute-active intervals rather than empty gaps in the command stream. Host "
    "launch overhead exists, but each launch controls enough work that its fixed cost is "
    "a small fraction of elapsed time. Increasing prompt length improves utilization until "
    "memory capacity or attention complexity becomes the limiting concern.\n\n"
    "Baseline decode execution\n"
    "Autoregressive decode has a different shape. Only one token is available at each "
    "iteration, and the next iteration cannot begin until the current logits are produced "
    "and a token is selected. Matrix operations collapse into narrow matrix-vector work. "
    "The baseline graph dispatches separate kernels for normalization, quantized weight "
    "unpacking, projections, rotary embedding, cache update, attention, residual paths, "
    "and feed-forward operations. A full token requires several hundred GPU commands. "
    "Many kernels run for only a few microseconds, comparable to submission, dependency, "
    "and synchronization costs. Timeline traces show bubbles between groups of commands, "
    "and tail work from one operation often leaves most execution units idle. Intermediate "
    "activations are written to global memory and read back by the next kernel even when "
    "the values are small enough to remain on chip in a fused implementation.\n\n"
    "Megakernel design\n"
    "The proposed path replaces most decode commands with one persistent GPU kernel. "
    "Workgroups cooperate through device-side barriers while moving through the transformer "
    "layers. Intermediate vectors remain in registers or local memory where practical. "
    "While one operation computes, asynchronous copies begin fetching weights and scales "
    "for the next operation. Specialized code handles normalization, quantized matrix-vector "
    "multiplication, rotary embedding, and residual updates without returning control to "
    "the host. Attention still reads the existing key-value cache from global memory, but "
    "cache update and surrounding element-wise work are fused. The final logits leave the "
    "kernel for token selection, so there remains one host-visible synchronization per "
    "generated token. Prefill continues to use the conventional graph because its larger "
    "matrix kernels are already efficient and do not fit the persistent execution scheme.\n\n"
    "Costs introduced by fusion\n"
    "Fusion is not free. The combined kernel has more live values, a larger instruction "
    "footprint, and more control flow than any baseline kernel. Register allocation rises "
    "from 64 registers per thread in the busiest baseline operation to 112 in the fused "
    "kernel. Consequently, only half as many workgroups can be resident on each subslice. "
    "Local-memory allocation also leaves little room for a second resident workgroup under "
    "some tuning configurations. If an asynchronous weight transfer misses its intended "
    "overlap window, reduced occupancy gives the scheduler fewer independent instructions "
    "with which to hide the stall. Device-wide barriers require every participating group "
    "to arrive, so load imbalance turns the slowest group into a critical path. Larger code "
    "can pressure the instruction cache, particularly where attention and feed-forward "
    "specializations alternate. These effects place an upper bound on useful fusion.\n\n"
    "Initial measurements\n"
    "Engineers collected results after ten warmup generations and thirty measured runs. "
    "With a 1024-token input and 128 generated tokens, the baseline reports a median time "
    "to first token of 386 ms and a median time per output token of 31.8 ms. The megakernel "
    "reports 381 ms and 19.6 ms respectively. The 99th-percentile token time falls from "
    "39.7 ms to 25.4 ms. For a 64-token input, time to first token changes from 71 ms to "
    "69 ms, while token time changes from 29.5 ms to 18.9 ms. For a 3072-token input, "
    "time to first token is about 1.31 seconds in both modes and token time rises to 23.7 ms "
    "for the megakernel because attention reads a larger cache. Package energy per generated "
    "token falls by 24 percent, although instantaneous GPU power is slightly higher.\n\n"
    "Timeline and counter evidence\n"
    "Command-stream traces attribute approximately 7.4 ms of each baseline decode step to "
    "gaps, command processing, and poorly occupied tails. Those categories shrink to less "
    "than 1 ms with the megakernel. Global activation traffic falls substantially, but total "
    "memory traffic falls by a smaller percentage because model weights must still be read. "
    "The fused path reaches 61 GB per second during its weight-intensive phases, close to "
    "the bandwidth observed when the display is active. Execution-unit occupancy declines "
    "from 58 percent in isolated baseline matrix kernels to 43 percent in the megakernel, "
    "yet useful work is spread more continuously over time. Barrier stalls account for "
    "roughly 6 percent of fused cycles. Experiments using a smaller register tile restore "
    "occupancy but perform more memory transactions and are slower overall. This suggests "
    "that occupancy alone is not an adequate optimization objective.\n\n"
    "Host-side processing\n"
    "The runtime returns logits to a CPU sampling stage after every iteration. Greedy "
    "decoding takes less than a millisecond, but production settings may apply temperature, "
    "top-p filtering, repetition penalties, grammar constraints, and stop-string checks. "
    "A Python-based integration once created enough CPU work to delay the next GPU submission "
    "by 5 ms, hiding much of the device improvement. Restricting the tensor library's thread "
    "pool reduced contention and restored the expected gain. The team must therefore report "
    "both device inference time and user-visible inter-token latency. It must also decide "
    "whether tokenization, detokenization, and streaming callbacks belong in the measured "
    "service-level metric. Optimizing the GPU cannot improve serial host work, so Amdahl's "
    "law becomes increasingly important as kernel execution gets faster.\n\n"
    "Correctness and reproducibility\n"
    "Both paths use the same model files, precision, tokenizer, prompts, and deterministic "
    "greedy generation. A preliminary comparison finds matching selected tokens for all "
    "short tests and cosine similarity above 0.999 for output logits. Longer generations "
    "occasionally diverge after a small numerical difference changes an argmax between two "
    "nearly equal candidates. Such divergence does not automatically indicate an invalid "
    "kernel, but it prevents direct latency comparisons if later tokens create different "
    "cache contents. A decode-only microbenchmark can hold token IDs and cache position "
    "fixed, while an end-to-end test should record generated IDs and flag the first mismatch. "
    "Compiled-model caches must be disabled or keyed by the megakernel setting; otherwise "
    "one run may silently reuse the other path's binary. Environment variables and plugin "
    "properties should be captured with every result.\n\n"
    "Potential experimental errors\n"
    "Several details could distort the result. Measuring the first invocation includes "
    "lazy compilation and memory allocation. Running all baseline samples before all fused "
    "samples can correlate one mode with a different thermal state. A changing cache length "
    "makes later decode steps more expensive than early ones, so fixed-position microbenchmarks "
    "and realistic growing-cache runs answer different questions. Comparing one-token "
    "generation with a long generation can mislabel sampling overhead as prefill. Background "
    "display activity, CPU frequency policy, and other processes add outliers. Means alone "
    "can obscure these effects. The test harness should randomize or alternate modes, pin "
    "relevant host work where appropriate, record temperatures and frequencies, and report "
    "medians plus tail percentiles. Confidence intervals are needed before treating small "
    "differences in time to first token as meaningful.\n\n"
    "Deployment tradeoffs\n"
    "The runtime team is considering three release choices. It can enable the megakernel "
    "for every supported decode shape, enable it only for batch one and selected context "
    "ranges, or retain the conventional graph until more devices are characterized. The "
    "first option is simple but risks regressions on GPUs with smaller register files or "
    "different barrier costs. The second requires a dispatch policy and trustworthy shape "
    "thresholds but preserves optimized prefill and may avoid pathological long-context "
    "cases. The third minimizes near-term risk but gives up a visible latency and energy "
    "improvement on the current product. Future models may use different hidden widths, "
    "mixture-of-experts layers, larger caches, or speculative decoding, so the policy should "
    "be based on measurable resource constraints rather than a hard-coded model name.\n\n"
    "Context-length scaling study\n"
    "A separate sweep holds generation length at 128 tokens and varies the cache position "
    "at which decode is measured. At 128 cached tokens, baseline and fused token times are "
    "29.1 ms and 18.4 ms. At 1024 positions they are 31.8 ms and 19.6 ms; at 4096 they are "
    "38.6 ms and 27.9 ms; and at 8192 they are 51.2 ms and 41.5 ms. The absolute launch and "
    "activation savings remain useful, but attention consumes a growing share of both paths "
    "as more keys and values must be read. Beyond 4096 positions, memory-controller counters "
    "remain near saturation for much of the fused iteration, leaving less opportunity to hide "
    "traffic with prefetching. The baseline also becomes more efficient in relative terms "
    "because its attention kernels run longer, amortizing their dispatch cost. The team has "
    "not yet measured whether compressed cache formats would shift this crossover point.\n\n"
    "Concurrency and scheduling\n"
    "Although today's interaction pattern is batch one, background summarization may later "
    "run beside an interactive request. The conventional runtime can interleave commands from "
    "two queues and prioritize latency-sensitive work at kernel boundaries. A persistent "
    "megakernel offers fewer preemption points and may monopolize execution resources for an "
    "entire token. Combining requests into a larger batch could improve baseline matrix "
    "utilization, reducing the relative value of fusion, while a fused kernel specialized for "
    "batch one might require recompilation or a separate variant. Continuous batching also "
    "introduces different cache lengths within one scheduling group and can worsen barrier "
    "imbalance. Any production policy must therefore include batch size and quality-of-service "
    "requirements, not just model identity and device name. Throughput under load should be "
    "measured separately from isolated request latency so one metric does not conceal harm to "
    "the other.\n\n"
    "Alternative optimizations considered\n"
    "The compiler team evaluated several less aggressive options. Fusing only adjacent "
    "element-wise operations removes some global-memory round trips but leaves most command "
    "gaps and improves token latency by only 9 percent. Capturing the baseline command sequence "
    "in a reusable device graph reduces host submission work and yields a 14 percent gain, but "
    "dependencies between small kernels still create poorly occupied tails. Increasing the "
    "number of in-flight requests raises throughput but violates the single-user latency goal. "
    "Weight compression below four bits lowers bandwidth demand, although early experiments "
    "show unacceptable accuracy loss on technical terminology. Moving sampling onto the GPU "
    "could remove the remaining host round trip, but it does not address per-layer launch costs "
    "and would complicate grammar-constrained decoding. These alternatives may complement the "
    "megakernel; they should not be treated as mutually exclusive without measuring combined "
    "resource use.\n\n"
    "Tuning sensitivity\n"
    "Three fused configurations were tested. The first favors large register tiles and achieves "
    "the best median latency on an idle device but loses 18 percent when display traffic is "
    "heavy. The second uses smaller tiles and more resident workgroups; it is 6 percent slower "
    "in the median yet has a tighter tail distribution under memory contention. The third "
    "allocates additional local memory for double-buffered weights, which helps short contexts "
    "but prevents enough groups from residing concurrently on this GPU. Compiler reports expose "
    "register count and scratch allocation, but they do not predict barrier imbalance or actual "
    "memory overlap. The selected variant should consequently be based on end-to-end measurements "
    "under realistic display and thermal conditions. A small win at laboratory boost frequency "
    "is less valuable than stable token pacing during a thirty-minute field session.\n\n"
    "Rollout and observability\n"
    "The application can ship both implementations and select one when the model is compiled. "
    "A guarded rollout would record anonymous aggregate counters for model shape, context bucket, "
    "selected path, compilation time, first-token latency, inter-token percentiles, device "
    "frequency, and fallback reason. It must not record prompt or generated text. A watchdog can "
    "fall back to the conventional graph after a compilation failure, unsupported shape, device "
    "reset, or correctness self-check failure. The policy should include a kill switch because "
    "driver revisions may alter register allocation or synchronization behavior. Offline release "
    "gates should test every supported device identifier, while field telemetry can reveal power "
    "states and concurrent workloads missing from the laboratory. Success means meeting latency "
    "and energy goals without increasing crashes, output mismatches, or long-tail stalls, rather "
    "than maximizing a single microbenchmark speedup.\n\n"
    "Requested analysis\n"
    "Write a concise engineering recommendation supported by the evidence above. Explain "
    "why fusion improves batch-one decode much more than prefill, separating launch overhead, "
    "on-chip intermediate reuse, weight traffic, and available parallelism. Identify at least "
    "three hardware or runtime limits that cap the speedup, and explain how register pressure, "
    "memory bandwidth, synchronization, cache growth, and host-side sampling interact rather "
    "than merely listing them. Interpret the supplied latency and counter measurements, noting "
    "which conclusions are strong and which remain uncertain. Propose a benchmark matrix that "
    "covers prompt length, generated length, cache position, thermal state, and sampling mode. "
    "Define correctness checks and statistical reporting requirements. Finally, recommend an "
    "enablement policy for this device and state what evidence would be required before using "
    "the same policy on a GPU with fewer execution units or a smaller register file."
)

PROMPTS: list[dict[str, str]] = [
    {
        "name": "short",
        "text": "What is the capital of France?",
    },
    {
        "name": "medium",
        "text": (
            "Explain, in a few sentences, how a transformer neural network uses "
            "self-attention to process a sequence of tokens, and why key-value "
            "caching makes autoregressive decoding faster than recomputing the "
            "whole sequence at every step."
        ),
    },
    {
        "name": "long",
        "text": (
            "You are a senior systems engineer. Read the following background "
            "carefully and then answer the question at the end.\n\n"
            "Large language models are deployed on a wide range of hardware, from "
            "small integrated GPUs to large data-center accelerators. During "
            "inference the model first runs a prefill phase that processes the "
            "entire prompt in a single forward pass, populating the key-value "
            "cache for every attention layer. After prefill the model enters the "
            "decode phase, generating one token at a time. Each decode step reads "
            "the growing key-value cache, computes attention against all previous "
            "tokens, and appends the new key and value vectors. Because the decode "
            "phase is memory-bandwidth bound and launches many small kernels, it "
            "often dominates end-to-end latency for long generations. A megakernel "
            "fuses the many small per-layer kernels of a decode step into a single "
            "GPU kernel launch, preloading weights for the next operation while the "
            "current one computes, using fine-grained synchronization, and removing "
            "kernel launch overhead and tail effects. This is particularly valuable "
            "for small models on small GPUs where launch overhead is a large "
            "fraction of the total step time.\n\n"
            "Question: Given the description above, explain why fusing the decode "
            "step into a single megakernel is expected to improve latency more than "
            "it improves prefill, and describe one hardware limitation that could "
            "reduce the achievable speedup on a small GPU."
        ),
    },
    {
        "name": "v. long",
        "text": VERY_LONG_PROMPT,
    },
]


def get_prompts(args) -> list[dict[str, str]]:
    """Prompt set for optimum/genai: a single user-supplied prompt when --prompt is
    given, otherwise the built-in short/medium/long trio."""
    custom = getattr(args, "prompt", None)
    if custom:
        return [{"name": "custom", "text": custom}]
    return PROMPTS


# ---------------------------------------------------------------------------
# OV inference helpers
# ---------------------------------------------------------------------------

def is_vlm_layout(model_dir) -> bool:
    """Multimodal exports (e.g. Gemma-4) ship a language model fed with
    inputs_embeds instead of openvino_model.xml."""
    return (Path(model_dir) / "openvino_language_model.xml").exists()


class Feeder:
    """Builds stateful-LM inputs for token ids, for both IR layouts.

    For the VLM layout the text embedding model runs on the same device to turn
    ids into inputs_embeds; it is only used outside the timed decode loop.
    """

    def __init__(self, core, model, model_dir: Path, device: str):
        self.names = {i.get_any_name() for i in model.inputs}
        self.embed = None
        if "inputs_embeds" in self.names:
            emb = core.compile_model(str(model_dir / "openvino_text_embeddings_model.xml"), device,
                                     {"CACHE_DIR": ""})
            self.embed = lambda ids: np.array(emb(ids.astype(np.int64))[0], dtype=np.float32)

    def __call__(self, ids: np.ndarray, past: int) -> dict[str, np.ndarray]:
        seq_len = ids.shape[1]
        total = past + seq_len
        feed = {
            "attention_mask": np.ones((BATCH, total), np.int64),
            "position_ids": np.arange(past, total, dtype=np.int64).reshape(1, seq_len),
            "beam_idx": np.zeros(BATCH, np.int32),
        }
        if self.embed is None:
            feed["input_ids"] = ids.astype(np.int64)
            return feed
        feed["inputs_embeds"] = self.embed(ids)
        if "token_type_ids" in self.names:
            feed["token_type_ids"] = np.zeros((BATCH, seq_len), np.int64)
        if "per_layer_inputs" in self.names:
            feed["per_layer_inputs"] = np.zeros((BATCH, seq_len, 1, 0), np.float32)
        return feed


def prefill_inputs(feeder: Feeder, input_ids: np.ndarray) -> dict[str, np.ndarray]:
    return feeder(input_ids, 0)


def single_token_inputs(feeder: Feeder, token_id: int, position: int) -> dict[str, np.ndarray]:
    """One-token step (decode, or one step of token-by-token priming)."""
    return feeder(np.array([[token_id]], np.int64), position)


def load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


def chat_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def prompt_token_ids(tokenizer, prompt: str) -> np.ndarray:
    text = chat_text(tokenizer, prompt)
    ids = tokenizer([text], return_tensors="np").input_ids
    return ids.astype(np.int64)


def stats(latencies_ms: list[float]) -> dict[str, float]:
    lat = sorted(latencies_ms)
    n = len(lat)
    mean = statistics.mean(lat)
    return {
        "mean": mean,
        "median": lat[n // 2],
        "min": lat[0],
        "p90": lat[int(0.90 * (n - 1))],
        "p99": lat[int(0.99 * (n - 1))],
        "tok_s": 1000.0 / mean,
        "count": n,
    }


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def decode_only_worker(args) -> list[dict]:
    """Pure decode benchmark using the OV native API.

    Prefill is run once (untimed) to warm the KV cache, then N identical
    single-token decode steps are timed at a fixed position.  No prefill
    latency is measured or reported.  A single fixed prompt is used; its
    content does not affect the measured decode cost.
    """
    import openvino as ov

    core = ov.Core()
    dev_name = core.get_property(args.device, "FULL_DEVICE_NAME")
    model_dir = Path(args.model_dir)
    ir = "openvino_language_model.xml" if is_vlm_layout(model_dir) else "openvino_model.xml"
    model = core.read_model(str(model_dir / ir))
    t0 = time.perf_counter()
    compiled = core.compile_model(model, args.device, {"CACHE_DIR": ""})
    compile_s = time.perf_counter() - t0
    feeder = Feeder(core, model, model_dir, args.device)

    tokenizer = load_tokenizer(model_dir)

    # Warmup – dummy prefill + a few decode steps
    warm = compiled.create_infer_request()
    warm.infer(prefill_inputs(feeder, np.ones((BATCH, 8), np.int64)))
    for pos in range(8, 12):
        warm.infer(single_token_inputs(feeder, 1, pos))

    # Tokenize the fixed prompt (content is irrelevant to the timed section)
    ids = prompt_token_ids(tokenizer, DECODE_ONLY_PROMPT)
    prompt_len = int(ids.shape[1])

    # Prefill to populate the KV cache (untimed)
    req = compiled.create_infer_request()
    res = req.infer(prefill_inputs(feeder, ids))
    logits = np.array(res[0])[0, -1, :].astype(np.float32)
    next_id = int(logits.argmax())

    # Optionally prime the KV cache to a longer context
    target_ctx = max(args.decode_ctx, prompt_len)
    for priming_pos in range(prompt_len, target_ctx):
        req.infer(single_token_inputs(feeder, next_id, priming_pos))

    decode_pos = target_ctx
    # The same decode input is reused every iteration to isolate kernel cost
    decode_input = single_token_inputs(feeder, next_id, decode_pos)

    for _ in range(args.warmup):
        req.infer(decode_input)
    lat = []
    for _ in range(args.tokens):
        t = time.perf_counter()
        req.infer(decode_input)
        lat.append((time.perf_counter() - t) * 1e3)

    # Greedy-decode real text for baseline vs megakernel comparison (untimed,
    # fresh request so the timed loop above is unaffected).
    text_out = _native_generate_text(compiled, feeder, tokenizer, ids, prompt_len, args.tokens)

    return [{
        "prompt": "decode",
        "prompt_len": prompt_len,
        "decode_ctx": decode_pos,
        "n_tok": args.tokens,
        "decode": stats(lat),
        "argmax": next_id,
        "logits": logits.tolist(),
        "device": dev_name,
        "compile_s": compile_s,
        "text": text_out,
    }]


def _native_generate_text(compiled, feeder: Feeder, tokenizer, ids: np.ndarray, prompt_len: int,
                           n_tokens: int) -> str:
    """Greedy-decode real tokens from a fresh request and detokenize."""
    gen_req = compiled.create_infer_request()
    r = gen_req.infer(prefill_inputs(feeder, ids))
    cur = int(np.array(r[0])[0, -1, :].argmax())
    pos = prompt_len
    # Gemma-4 ends a turn with <end_of_turn> (id 106), which is listed in
    # generation_config.json but is not the tokenizer's eos_token.
    eos = {tokenizer.eos_token_id} - {None}
    gen_cfg = Path(tokenizer.name_or_path) / "generation_config.json"
    if gen_cfg.exists():
        ids_cfg = json.loads(gen_cfg.read_text()).get("eos_token_id", [])
        eos |= set(ids_cfg if isinstance(ids_cfg, list) else [ids_cfg])
    gen_ids: list[int] = []
    for _ in range(n_tokens):
        gen_ids.append(cur)
        if cur in eos:
            break
        r = gen_req.infer(single_token_inputs(feeder, cur, pos))
        pos += 1
        cur = int(np.array(r[0])[0, -1, :].argmax())
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def optimum_worker(args) -> list[dict]:
    import torch
    from optimum.intel import OVModelForCausalLM, OVModelForVisualCausalLM
    from transformers import AutoTokenizer

    # Cap torch's intra-op threads. optimum's generate() runs its sampling/
    # logits-processing (torch isin/where/arange, all multi-threaded) on the host
    # BETWEEN forwards, while OpenVINO's async GPU infer path keeps host threads
    # busy. With torch defaulting to one thread per core, those gap ops
    # oversubscribe the CPU and get throttled 2-3x -- which erases the
    # MegaKernel's faster inference and makes decode look no faster than baseline.
    # Leaving CPU headroom removes the oversubscription and restores the speedup.
    n_threads = args.torch_threads or max(1, (os.cpu_count() or 8) // 4)
    torch.set_num_threads(n_threads)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    t0 = time.perf_counter()
    # CACHE_DIR="" disables the compiled-model blob cache. The cache key does not
    # include the OV_MEGAKERNEL_DISABLE env var, so leaving it on would make the
    # MegaKernel run silently reuse the baseline blob (no transformation).
    model_cls = OVModelForVisualCausalLM if is_vlm_layout(args.model_dir) else OVModelForCausalLM
    model = model_cls.from_pretrained(
        args.model_dir, device=args.device, ov_config={"CACHE_DIR": ""})
    compile_s = time.perf_counter() - t0

    import openvino as ov
    dev_name = ov.Core().get_property(args.device, "FULL_DEVICE_NAME")

    def gen(model_inputs, n_new):
        t = time.perf_counter()
        out = model.generate(
            **model_inputs,
            max_new_tokens=n_new,
            min_new_tokens=n_new,
            do_sample=False,
            num_beams=1,
        )
        return (time.perf_counter() - t) * 1e3, out

    results = []
    for prompt in get_prompts(args):
        text = chat_text(tokenizer, prompt["text"])
        model_inputs = tokenizer([text], return_tensors="pt")
        prompt_len = int(model_inputs.input_ids.shape[1])

        # warmup
        for _ in range(max(1, args.gen_warmup)):
            gen(model_inputs, args.tokens)

        # TTFT (prefill only) = generate exactly one new token.
        ttft = []
        for _ in range(args.gen_iters):
            ms, _ = gen(model_inputs, 1)
            ttft.append(ms)
        # Full generation of args.tokens tokens.
        full = []
        last_out = None
        for _ in range(args.gen_iters):
            ms, last_out = gen(model_inputs, args.tokens)
            full.append(ms)

        ttft_mean = statistics.mean(ttft)
        full_mean = statistics.mean(full)
        out_ids = last_out[0][prompt_len:].tolist()
        actual_n_tok = len(out_ids)
        # TTFT measures 1-token generation; remaining (actual_n_tok - 1) tokens
        # are decode steps. Guard against degenerate short outputs.
        n_decode = max(actual_n_tok - 1, 1)
        decode_total = max(full_mean - ttft_mean, 1e-6)
        per_tok_ms = decode_total / n_decode
        gen_text = tokenizer.decode(out_ids, skip_special_tokens=True)

        results.append({
            "prompt": prompt["name"],
            "prompt_len": prompt_len,
            "prefill_ms": ttft_mean,
            "n_tok": actual_n_tok,
            "decode": {
                "mean": per_tok_ms,
                "median": per_tok_ms,
                "tok_s": 1000.0 / per_tok_ms,
                "count": n_decode,
            },
            "device": dev_name,
            "compile_s": compile_s,
            "text": gen_text,
        })
    return results


def genai_worker(args) -> list[dict]:
    import openvino_genai as ov_genai
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    t0 = time.perf_counter()
    # CACHE_DIR="" disables the compiled-model blob cache (see optimum_worker).
    pipeline_kwargs: dict = {"CACHE_DIR": ""}
    pipeline_kwargs["ATTENTION_BACKEND"] = "SDPA"
    vlm = is_vlm_layout(args.model_dir)
    pipe_cls = ov_genai.VLMPipeline if vlm else ov_genai.LLMPipeline
    pipe = pipe_cls(args.model_dir, args.device, **pipeline_kwargs)
    compile_s = time.perf_counter() - t0

    import openvino as ov
    dev_name = ov.Core().get_property(args.device, "FULL_DEVICE_NAME")

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = args.tokens
    cfg.min_new_tokens = args.tokens
    cfg.do_sample = False
    cfg.num_beams = 1
    cfg.apply_chat_template = False

    def run(text):
        return pipe.generate(text, generation_config=cfg) if vlm else pipe.generate([text], cfg)

    results = []
    for prompt in get_prompts(args):
        text = chat_text(tokenizer, prompt["text"])
        prompt_len = int(tokenizer([text], return_tensors="np").input_ids.shape[1])

        for _ in range(max(1, args.gen_warmup)):
            run(text)

        ttft_ms, tpot_ms, tput = [], [], []
        gen_text = ""
        generated_texts = set()
        for _ in range(args.gen_iters):
            res = run(text)
            pm = res.perf_metrics
            actual_prompt_len = pm.get_num_input_tokens()
            if actual_prompt_len != prompt_len:
                raise RuntimeError(
                    f"GenAI input token count differs for {prompt['name']}: "
                    f"expected {prompt_len}, got {actual_prompt_len}"
                )
            ttft_ms.append(pm.get_ttft().mean)
            tpot_ms.append(pm.get_tpot().mean)     # decode: mean ms / output token
            tput.append(pm.get_throughput().mean)
            gen_text = res.texts[0] if getattr(res, "texts", None) else str(res)
            generated_texts.add(gen_text)

        tpot_mean = statistics.mean(tpot_ms)
        actual_n_tok = res.perf_metrics.get_num_generated_tokens()
        results.append({
            "prompt": prompt["name"],
            "unique_outputs": len(generated_texts),
            "prompt_len": prompt_len,
            "prefill_ms": statistics.mean(ttft_ms),
            "n_tok": actual_n_tok,
            "decode": {
                "mean": tpot_mean,
                "median": tpot_mean,
                "tok_s": 1000.0 / tpot_mean,
                "throughput_tok_s": statistics.mean(tput),
                "count": actual_n_tok,
            },
            "device": dev_name,
            "compile_s": compile_s,
            "text": gen_text,
        })
    return results


WORKERS = {
    "decode_only": decode_only_worker,
    "optimum": optimum_worker,
    "genai": genai_worker,
}


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def worker_env(framework: str, path: str) -> dict:
    env = os.environ.copy()
    env["OV_MEGAKERNEL_DISABLE"] = "0" if path == "megakernel" else "1"
    return env


def spawn(framework: str, path: str, args) -> list[dict]:
    cmd = [
        sys.executable, __file__, "--worker", framework, "--path", path,
        "--model-dir", str(args.model_dir), "--device", args.device,
        "--warmup", str(args.warmup), "--tokens", str(args.tokens),
        "--decode-ctx", str(args.decode_ctx),
        "--torch-threads", str(args.torch_threads),
        "--gen-warmup", str(args.gen_warmup), "--gen-iters", str(args.gen_iters),
    ]
    if getattr(args, "prompt", None):
        cmd += ["--prompt", args.prompt]
    out = subprocess.run(cmd, env=worker_env(framework, path), capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout)
        sys.stderr.write(out.stderr)
        raise RuntimeError(f"{framework}/{path} worker failed (exit {out.returncode})")
    lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
    for ln in lines[:-1]:
        print(f"    [{framework}/{path}] {ln}")
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def _maybe_print_text(base: list[dict], mega: list[dict]) -> None:
    """Print generated text only when baseline and megakernel outputs differ."""
    for b, m in zip(base, mega):
        for path, result in (("baseline", b), ("megakernel", m)):
            if result.get("unique_outputs", 1) > 1:
                print(f"  *** NON-REPEATABLE OUTPUT [{b['prompt']}/{path}]: "
                      f"{result['unique_outputs']} distinct measured outputs ***")
        bt = (b.get("text") or "").strip()
        mt = (m.get("text") or "").strip()
        if bt != mt:
            print(f"  *** OUTPUT MISMATCH [{b['prompt']}] ***")
            print(f"    baseline  : {bt!r}")
            print(f"    megakernel: {mt!r}")


def print_decode_only_table(base: list[dict], mega: list[dict]) -> None:
    W = 86
    print()
    print("=" * W)
    print(" DECODE-ONLY  (OV native API)")
    print("=" * W)
    print(" How it works:")
    print("   Prefill is executed once (untimed) to warm the KV cache.")
    print("   Then N identical single-token decode steps are timed at a fixed")
    print("   KV-cache position.  No prefill latency is measured or reported.")
    print("   Prompt content does not affect the measured decode cost.")
    print("   decode_x is the per-token decode speedup (primary metric).")
    print()

    hdr = (f"  {'ctx':>5}  {'n_tok':>5} | "
           f"{'base ms/tok':>11}  {'base tok/s':>10} | "
           f"{'mk ms/tok':>9}  {'mk tok/s':>8} | "
           f"{'decode_x':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for b, m in zip(base, mega):
        bd = b["decode"]["mean"]
        md = m["decode"]["mean"]
        b_toks = b["decode"]["tok_s"]
        m_toks = m["decode"]["tok_s"]
        dec_x = bd / md if md else float("nan")
        extra = ""
        if "logits" in b and "logits" in m:
            match = b["argmax"] == m["argmax"]
            cos = cosine(b["logits"], m["logits"])
            extra = f"   argmatch={match}  cos={cos:.4f}"
        print(f"  {b['decode_ctx']:>5}  {b['n_tok']:>5} | "
              f"{bd:>11.3f}  {b_toks:>10.1f} | "
              f"{md:>9.3f}  {m_toks:>8.1f} | "
              f"{dec_x:>7.2f}x{extra}")

    _maybe_print_text(base, mega)


def print_optimum_table(base: list[dict], mega: list[dict], n_tokens: int) -> None:
    W = 98
    print()
    print("=" * W)
    print(" OPTIMUM-INTEL  (HF OVModelForCausalLM.generate)")
    print("=" * W)
    print(" How it works:")
    print("   TTFT = time to first token (prefill latency), measured by generating")
    print("   exactly 1 new token.  ms/tok = per-decode-token latency estimated as")
    print("   (full_generate_ms - ttft_ms) / (n_generated - 1), averaged over")
    print("   multiple generate() calls.  decode_x is the primary speedup metric.")
    print()

    hdr = (f"  {'prompt':<8}  {'in_tok':>6} | "
           f"{'base ttft':>9}  {'mk ttft':>7} | "
           f"{'base ms/tok':>11}  {'mk ms/tok':>9} | "
           f"{'decode_x':>8}  {'pf_x':>5}  {f'e2e_x@{n_tokens}':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for b, m in zip(base, mega):
        bd = b["decode"]["mean"]
        md = m["decode"]["mean"]
        bpf = b.get("prefill_ms", float("nan"))
        mpf = m.get("prefill_ms", float("nan"))
        dec_x = bd / md if md else float("nan")
        pf_x = bpf / mpf if mpf else float("nan")
        n_dec = max(n_tokens - 1, 0)
        e2e = (bpf + n_dec * bd) / (mpf + n_dec * md) if mpf and md else float("nan")
        print(f"  {b['prompt']:<8}  {b['prompt_len']:>6} | "
              f"{bpf:>9.3f}  {mpf:>7.3f} | "
              f"{bd:>11.3f}  {md:>9.3f} | "
              f"{dec_x:>7.2f}x  {pf_x:>4.2f}x  {e2e:>9.2f}x")

    _maybe_print_text(base, mega)


def print_genai_table(base: list[dict], mega: list[dict], n_tokens: int) -> None:
    W = 98
    print()
    print("=" * W)
    print(" OPENVINO GENAI  (ov_genai.LLMPipeline.generate)")
    print("=" * W)
    print(" How it works:")
    print("   TTFT and TPOT come from ov_genai perf_metrics, averaged over")
    print("   multiple generate() calls.  decode_x is the primary speedup metric.")
    print()

    hdr = (f"  {'prompt':<8}  {'in_tok':>6} | "
           f"{'base ttft':>9}  {'mk ttft':>7} | "
           f"{'base ms/tok':>11}  {'mk ms/tok':>9} | "
           f"{'decode_x':>8}  {'pf_x':>5}  {f'e2e_x@{n_tokens}':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for b, m in zip(base, mega):
        bd = b["decode"]["mean"]
        md = m["decode"]["mean"]
        bpf = b.get("prefill_ms", float("nan"))
        mpf = m.get("prefill_ms", float("nan"))
        dec_x = bd / md if md else float("nan")
        pf_x = bpf / mpf if mpf else float("nan")
        n_dec = max(n_tokens - 1, 0)
        e2e = (bpf + n_dec * bd) / (mpf + n_dec * md) if mpf and md else float("nan")
        print(f"  {b['prompt']:<8}  {b['prompt_len']:>6} | "
              f"{bpf:>9.3f}  {mpf:>7.3f} | "
              f"{bd:>11.3f}  {md:>9.3f} | "
              f"{dec_x:>7.2f}x  {pf_x:>4.2f}x  {e2e:>9.2f}x")

    _maybe_print_text(base, mega)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--device", default=DEFAULT_DEVICE, help="GPU.1 = B60 dGPU")
    ap.add_argument("--frameworks", nargs="+", default=["decode_only", "optimum", "genai"],
                    choices=list(WORKERS))
    ap.add_argument("--only-framework", choices=list(WORKERS), default=None,
                    help="Run a single framework (overrides --frameworks).")
    ap.add_argument("--tokens", type=int, default=150,
                    help="Decode steps to time (decode_only) / tokens to generate "
                         "(optimum, genai) per path.")
    # decode_only tuning
    ap.add_argument("--warmup", type=int, default=5,
                    help="Decode-step warmup iterations before timing (decode_only only).")
    ap.add_argument("--decode-ctx", type=int, default=0,
                    help="If >0, prime the KV cache to this length before timing decode "
                         "(isolates kernel cost from O(context) attention growth). "
                         "decode_only path only.")
    # optimum / genai generate() benchmark
    ap.add_argument("--torch-threads", type=int, default=23,
                    help="Cap torch intra-op threads in the optimum path "
                         "(0 = auto: ~cores/4).")
    ap.add_argument("--gen-warmup", type=int, default=1,
                    help="generate() warmup calls (optimum/genai paths).")
    ap.add_argument("--gen-iters", type=int, default=3,
                    help="generate() measured calls (optimum/genai paths).")
    ap.add_argument("--prompt", default=None,
                    help="Single custom prompt for optimum/genai paths (overrides the "
                         "built-in short/medium/long trio). Has no effect on decode_only.")
    # internal
    ap.add_argument("--worker", choices=list(WORKERS), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--path", choices=("baseline", "megakernel"), default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        args.model_dir = str(args.model_dir)
        results = WORKERS[args.worker](args)
        print(json.dumps(results))
        return

    frameworks = [args.only_framework] if args.only_framework else args.frameworks
    print(f"Device: {args.device}   frameworks: {frameworks}")
    print(f"tokens={args.tokens}   decode_only warmup={args.warmup}   "
          f"generate: warmup={args.gen_warmup} iters={args.gen_iters}")

    all_results: dict[str, dict[str, list[dict]]] = {}
    for fw in frameworks:
        all_results[fw] = {}
        for path in ("baseline", "megakernel"):
            print(f"\n>>> running {fw}/{path} ...")
            all_results[fw][path] = spawn(fw, path, args)

    for fw in frameworks:
        base = all_results[fw]["baseline"]
        mega = all_results[fw]["megakernel"]
        if fw == "decode_only":
            print_decode_only_table(base, mega)
        elif fw == "optimum":
            print_optimum_table(base, mega, args.tokens)
        elif fw == "genai":
            print_genai_table(base, mega, args.tokens)

    print()


if __name__ == "__main__":
    main()
