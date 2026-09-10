# Debugging Accuracy Issues in the GGUF Frontend / OpenVINO Backend

How to find *why* a GGUF model produces wrong output through the OpenVINO frontend +
`ggml-openvino` backend, versus the reference llama.cpp CPU path.

A field guide, not a spec: the techniques that have paid off and the traps that wasted
time. Read it before reaching for a debugger.

---

## The one rule: always have an authoritative reference

Every accuracy claim is a comparison, and the reference must be **the real llama.cpp CPU
implementation** — never one you derived by hand. A hand-written reference (numpy, a C++
recurrence from the paper) encodes *your* understanding of the op's layout; if that is wrong
the same way the frontend is, the two agree and the bug hides — or, worse, a correct frontend
"fails" against your buggy reference and you chase a phantom.

If you catch yourself typing out an op's math to build an expectation, stop and generate it
from ggml instead (see [The ggml-CPU oracle](#the-ggml-cpu-oracle)).

> Trap: with one head (`H=1`) many layout orders coincide, so a single-head unit test can pass
> against a wrong reference and give false confidence. Test at real dimensions (step 9).

---

## Bisection strategy, coarse to fine

Work from the cheapest, coarsest signal to the finest; each step narrows the search space for
the next. Don't open a debugger until these have cornered the bug.

### 1. Is it an OpenVINO bug at all?

When the output *text* is bad but the graph runs end-to-end, first decide whether OV is even at
fault, by comparing against a genuine ggml-CPU run of the **same model at the same commit**.

> ⚠️ **A build with `-DGGML_OPENVINO=ON` has no CPU backend.** The OV backend registers as a
> ggml *device* (`OPENVINO0`); no CPU device is registered (`llama-cli --list-devices` shows
> only `OPENVINO0`). So every "CPU" knob silently runs OV: `env -u GGML_OPENVINO` and
> `GGML_OPENVINO=""` do nothing (it's a device, not gated by that var), `--device none` runs
> OV, and `GGML_OPENVINO_DEVICE=CPU` runs OV-on-CPU. "I compared CPU vs OV and they matched" is
> really OV vs OV — a conclusion that has wasted days.

The only reliable CPU reference is a second build with OV compiled out, at the same commit:

```
cmake -B build-cpu -DGGML_OPENVINO=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build-cpu --target llama-simple llama-eval-callback -j$(nproc)
```

Compare greedy on **`llama-simple`** (raw completion, no chat template — see the gotcha below):

```
./build-cpu/bin/llama-simple -m <model>.gguf -n 20 "The capital of France is"   # true ggml CPU
./build-ov/bin/llama-simple  -m <model>.gguf -n 20 "The capital of France is"   # OpenVINO
```

- **Both good** → port correct, done.
- **CPU good, OV bad** → a genuine OV bug; localize with the steps below against `build-cpu`.
- **Both bad** → only now may it be a shared llama.cpp/model/template issue; confirm the CPU
  build is genuinely OV-free (`--list-devices`) before believing it.

> **The chat template hides the signal.** `llama-cli`/`llama-completion` apply the model's chat
> template; a thinking model then emits `<think>` and reasons instead of completing, so greedy
> output looks degenerate even when the math is fine. Use `llama-simple` for bisection; reserve
> `llama-cli` for judging end-user quality *after* the math is verified.

### 2. Graph bug or quantization?

Run the **same quantized weights** through the CPU reference and OV, both greedy, and compare
the first token. Reference right + OV wrong on identical weights ⇒ quantization is ruled out,
it's a graph/conversion bug. Don't download a higher-precision model to "check quantization" —
this already answered it for free (and big models often OOM in the frontend; see Gotchas).

### 3. Prefill or decode?

`-n 1` (prefill only) vs `-n 12` (prefill + decode). First token already wrong → the bug is in
the prefill graph; ignore all state/KV/decode machinery. Prefill right, later tokens drift →
suspect stateful bookkeeping, the KV/recurrent-state round-trip, or a decode-only path.

### 4. First-divergence diff (highest-yield localization)

Dump every node's output on both backends and diff in graph order. `llama-eval-callback` fires
per ggml node and prints name, op, shapes and a per-tensor **`sum`** — a cheap, position-stable
fingerprint. Node names repeat across layers/experts, so align the two logs **positionally**,
not by name.

```
./build-cpu/bin/llama-eval-callback -m <model>.gguf -p "The capital of France is" -n 1 > /tmp/cb_ref.log 2>&1
GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=1 \
    ./build-ov/bin/llama-eval-callback -m <model>.gguf -p "The capital of France is" -n 1 > /tmp/cb_ov.log 2>&1

awk '/cb_eval:/{n=$2} /sum = /{print n"\t"$3}' /tmp/cb_ref.log > /tmp/ref_pos.txt
awk '/cb_eval:/{n=$2} /sum = /{print n"\t"$3}' /tmp/cb_ov.log  > /tmp/ov_pos.txt
paste /tmp/ov_pos.txt /tmp/ref_pos.txt | awk -F'\t' '{
  d=$2-$4; if(d<0)d=-d; b=($4<0?-$4:$4); if(b<1)b=1; rel=d/b;
  flag=($1!=$3)?" NAME-MISMATCH":(rel>0.001?" <<< DIFF":"");
  printf "%3d %-26s ov=%-12s ref=%-12s%s\n", NR,$1,$2,$4,flag }'
```

Judgement calls, all learned the hard way:

- **The first *visible* divergence is usually not the root cause** — often benign kernel
  rounding (a sub-0.2% matmul diff). Confirm a candidate by *eliminating* it (steps 6–7) and
  seeing the output change; never assume.
- **Only tensors at subgraph boundaries carry real data** (see the trustworthy-tensor rule in
  step 5); interior nodes read `0` or stale, so a zero-vs-nonzero "difference" there is an
  artifact.
- **The callback perturbs OV's graph split** (it forces boundaries) and may crash OV early, but
  the prefix is usually enough to find the *first* divergence. For a full-depth sweep that
  doesn't perturb the split, use `GGML_OPENVINO_DEBUG_OUTPUT` (step 5).
- **Positional alignment holds only until node order/count diverges** between backends. Past the
  first genuine name mismatch, trust only rows where both names still agree.

### 5. Per-layer sweep — element-wise cosine, never Min/Max/Mean

> ⚠️ **A permutation-invariant statistic cannot see a permutation bug.** `sum`, mean, min and
> max are unchanged when a tensor's elements are *reordered*. A whole class of bugs only reorders
> correct values — partial-rotary rope rotating the wrong slice, a head-scramble from a bad
> reshape/transpose, interleaved-vs-split-halves, any layout error on an axis a size-1 dim would
> collapse. A statistic-based per-layer sweep then reports "small diffuse drift, no single broken
> layer" — a false verdict that has sent debugging down a precision rabbit hole for a day.
> **Never conclude "drift" from a permutation-invariant metric.**

`GGML_OPENVINO_DEBUG_OUTPUT=1` prints First/Min/Max/Mean per *subgraph output* at its natural
boundary (no split perturbation, no early crash), covering every layer to `result_output`. But
compute **element-wise cosine** against the reference, not the printed statistics:

```
cos = dot(ov, ref) / (norm(ov) * norm(ref))
```

Cosine is ~1.0 when the buffers agree elementwise and collapses the instant elements are
reordered. Sweep it per layer on full-width, same-named, genuine-boundary tensors. The shape of
the curve *is* the diagnosis:

- **A sharp cliff** at one layer (e.g. 0.999997 → 0.43 between neighbours) = a **localized
  structural bug** there. Align the cliff to what that layer does differently (e.g. "first layer
  downstream of a rope layer") to point at the op.
- **A gentle monotone slope** from ~1.0 = genuine accumulating precision drift — a verdict you
  earn only when cosine *also* shows a smooth slope.

To place both backends at every layer despite OV crashing early under the eval-callback, match
the complete ggml-CPU eval-callback `sum` per layer (÷ element count) against the OV
`DEBUG_OUTPUT` mean for the same layer: a small, one-signed, compounding diff with no single jump
is a mildly biased op applied every layer; a single jump is a localized bug. (To gauge severity,
bias logits with `-l <TOKEN_ID>±N` and bisect N until top-1 flips — a tight reference margin but
a large OV displacement means real distortion, not a rounding-broken tie.)

> **Trustworthy-tensor rule (applies to steps 4, 5, 8).** ggml reuses scratch buffers, so an
> interior tensor promoted to an output can read memory already clobbered by a later op — you get
> plausible garbage. Only **persistent buffers** are safe to compare cross-backend: the
> KV/recurrent-state caches (`cache_k/v/r/s`) and genuine subgraph-boundary outputs. A pure
> `TRANSPOSE` that appears to change its element `sum` is diagnostic *of this artifact*, not of a
> transpose bug. Promote at most **one** interior tensor per run; several at once perturb
> allocation and corrupt each other. Confirming a cache write matches the reference also *bounds*
> the bug: everything feeding it is correct, so the fault is downstream.

### 6. Quant-path isolation — `GGML_OPENVINO_DISABLE_TYPES`

Forcing a *type* to CPU is surgical: it moves only that quant type's tensors to ggml while
leaving graph structure intact (unlike forcing an op, step 7). Two high-value moves:

- Disable **one** suspect type to test whether that quant kernel is the culprit.
- Disable **all** materialized quant types at once
  (`Q4_0,Q4_1,Q4_K,Q5_K,Q8_0,Q6_K,Q5_1,MXFP4`): every weight matmul then runs on ggml with
  bit-identical weights and OV executes only the f32/f16 structural ops. Still wrong ⇒ the bug is
  **structural**, every quant kernel decisively ruled out — a narrowing no single op-force gives.

### 7. Op-family isolation — `GGML_OPENVINO_DISABLE_OPS`

`ggml_backend_openvino_device_supports_op()` decides per op whether OV claims it or ggml's CPU
kernel runs. `GGML_OPENVINO_DISABLE_OPS=SSM_CONV,DIV` forces the listed ops to CPU; binary-search
the suspect set. For a novel arch, start with the arch-specific ops (shared ops are already
exercised by working models).

- **Output correct** → that op (or its glue) was the bug.
- **Output unchanged** → op exonerated.
- **New crash / shape error** → *inconclusive*, not guilt: forcing an op re-splits the graph and
  can surface an unrelated boundary bug. Try another combination, or go to the op-level test.

### 8. Pinpoint a layout / wrong-source bug — full-element dumps, multiset, distribution matching

A cosine cliff (step 5) says *where* the graph scrambles and *that* the bug is a permutation, not
arithmetic. It doesn't say *what* the wrong tensor is a permutation *of* — usually the whole
answer, because a scramble means the op read the **wrong source region** (a neighbouring head,
the sibling half of a joint projection, a strided window read as contiguous).

Dump full element order (not statistics) on both backends:

- OV: `GGML_OPENVINO_DUMP_TENSOR=<name-substr>` → `/tmp/ov_dump_*`, one float per line.
- CPU: `GGML_DUMP_TENSOR=<name-substr>` (ggml eval-callback) → `/tmp/ref_dump_*`.

Then two comparisons in order:

1. **Multiset (sorted) comparison — layout vs math.** Sort both dumps' nonzero values and
   compare. Sorted-match ≈ 1.0 but position-wise cosine low ⇒ the numbers are right, only
   positions wrong: a pure layout/read bug — look at layout, not any kernel's math. Sorted-match
   also low ⇒ the values differ: a real arithmetic bug — back to steps 6–7 and the oracle.
2. **Distribution matching against multiple candidate references — the wrong-source finder.**
   When step 1 says "right numbers, wrong place," the values came from *somewhere*. Dump several
   candidate reference tensors near the suspect (the sibling projection half, the adjacent head,
   the pre-/post-transform version) and compare the OV tensor's distribution (sorted multiset;
   range/mean/std as a quick fingerprint) to each. "OV's X matches the reference's *Y*, not X"
   names the region the op wrongly grabbed. (Obey the trustworthy-tensor rule.)

Then read the offending op's layout directly. Very often it's a `VIEW`/reshape: dump its ggml
geometry (`ne[]`, `nb[]` strides, offset) and the `op_case` the decoder assigned. A view whose
source is strided/gapped (stride ≠ extent, a sub-block of a larger group) but classified as a
*dense contiguous* window is the bug — the contiguous handler slices through the gap and
scrambles elements. Fix the **classification** (a stride/density check that routes the strided
case to a stride-aware handler), not the consumer. `GGML_OPENVINO_DUMP_CGRAPH=1` gives `ne[]`; a
temporary `getenv`-gated `fprintf` at the classification site gives `nb[]`/offset/op_case.

### 9. Reproduce at the op level with a ggml oracle

Once an op is implicated, stop testing through the full model and reproduce it in the frontend's
unit tests (`tests/test_ops.cpp`), which build a one-op `ov::Model` via `SingleOpBuilder` and run
on CPU in milliseconds. This is where the bug is pinned and the regression test lives.

Feed **real model dimensions**, not `1×1×1` — most layout bugs live on an axis a size-1 dim
collapses: multi-head (`H>1`), GQA repeat (`H_v ≠ H_k`), multi-token (`T>1`), batch (`B>1`). Pull
the real shapes from a cgraph dump (`GGML_OPENVINO_DUMP_CGRAPH=1`). A test that passes at `H=1`
proves nothing about a model that uses `H=32`.

---

## The ggml-CPU oracle

For ground-truth op values, build a tiny C program that links `libggml` + `libggml-cpu`, builds a
one-node graph, runs it on CPU, and prints the output — those numbers become the `expected`
array. `tests/gdn_oracle.c` is a worked example; copy its structure.

```
gcc <op>_oracle.c -o <op>_oracle \
    -I <llama.cpp>/ggml/include \
    -L <llama.cpp>/build-ov/bin -lggml -lggml-base -lggml-cpu -lm \
    -Wl,-rpath,<llama.cpp>/build-ov/bin
LD_LIBRARY_PATH=<llama.cpp>/build-ov/bin ./<op>_oracle
```

- **Link against the same `libggml` the backend uses.** A stale tree can assert on a different
  shape than the source you're reading.
- Set `no_alloc = true` when using `ggml_backend_alloc_ctx_tensors`.
- Read the op's shape contract from the **header** (`ggml.h`); don't guess `ne[]` order.
- Fill inputs with distinct, asymmetric values so a transposed/mis-strided axis changes a number.
- If the op has more than one conversion path (e.g. a fused internal op and a decomposed Loop
  fallback), assert **both** against the oracle — matching a shared oracle proves both correct.

---

## Bug archetypes

Concrete structural bugs seen so far, each with the generalizable lesson. When a cosine cliff or
a dynamic-shape crash points near one, check it first.

**A. Geometry read from the tensor instead of the config.** Partial-rotary rope: a head is 256
wide but only the first 64 dims rotate; taking the rotary width from the tensor's last dim (256)
rotates the pass-through tail and corrupts every layer using that op. *Lesson: an op must read
its geometry (rotary width, head count) from the model config, not a tensor dim — the two differ
whenever a dimension is partial. A test at full width can't catch it.*

**B. A shared precomputed table built with the wrong config flag.** A sin/cos table computed once
and shared across layers must be built with the same per-op config its consumers assume; a mode
flag defaulting wrong makes the table's token axis mismatch the data and broadcast-crashes at the
consuming `Multiply`. *Lesson: a shared/precomputed value must be built with the consumer's
config; the symptom is a shape/broadcast mismatch, not a wrong number.*

**C. A dynamic-axis slice with a baked absolute offset.** A tail slice (the last *k* columns of a
window whose length varies prefill vs decode) emitted with an absolute start computed at prefill
over-reads at decode. *Lesson: anchor a slice to the end it's pinned to — negative start, open
end — never to an offset baked at prefill.*

**D. A strided VIEW mis-classified as contiguous.** An interleaved layout
(`[A_h0, B_h0, A_h1, B_h1, …]`) gives a per-`A` view a group stride larger than its extent;
classified as a dense window, the contiguous handler slices through the gap and scrambles heads.
*Lesson: a VIEW's handler must be chosen from its strides (`nb[]`), not its element count — two
views with identical shape and offset but different strides need different handlers.*

> Meta-pattern behind all four: a value *correct for prefill / the common case* (full head, one
> rope mode, prefill length, a contiguous layout) is hard-coded instead of read from the true
> per-op/per-step geometry, so it silently breaks at decode or on a different architecture.

> **After any shared-path fix — especially a VIEW/`op_case` predicate — re-run the other
> supported models and confirm their classification/output is unchanged.** A stride/density guard
> that fixes arch A can reject arch B's legitimately-contiguous view.

---

## Debug env vars

Backend seams live in `ggml/src/ggml-openvino/`. Gate any new debug output on a `getenv` check
and write to `stderr`.

| Env var | Effect |
|---|---|
| `GGML_OPENVINO_DEVICE=CPU` | Run OV on CPU. (Still OV, not ggml — see step 1.) |
| `GGML_OPENVINO_STATEFUL_EXECUTION=1` | Enable the stateful KV/recurrent-state path (real decode flow). |
| `GGML_OPENVINO_DUMP_CGRAPH=1` | Dump each OV subgraph as `cgraph_ov_N.txt` — ops, names, per-tensor `ne[]`. Primary tool for real op dimensions and how the graph split. |
| `GGML_OPENVINO_DUMP_IR=1` | Serialize each compiled subgraph to `model_*.xml`. Unavailable for models with internal (non-serializable) ops. |
| `GGML_OPENVINO_DEBUG_INPUT=1` / `_DEBUG_OUTPUT=1` | Print each bound input / output tensor (name, shape, First/Min/Max/Mean). |
| `GGML_OPENVINO_DUMP_TENSOR=<substr>` | Dump **full element order** of each matching OV output to `/tmp/ov_dump_*` (one float/line) — for the position-wise + multiset compare of step 8; not permutation-invariant. |
| `GGML_DUMP_TENSOR=<substr>` (ggml `common/debug.cpp`) | CPU-reference counterpart: matching tensors from the ggml eval-callback, in element order. |
| `GGML_OPENVINO_DISABLE_OPS=SSM_CONV,DIV` | Force listed ggml ops to CPU (step 7). Re-splits the graph — a new crash is inconclusive. |
| `GGML_OPENVINO_DISABLE_TYPES=Q8_0,Q6_K` | Force listed quant types to CPU without changing graph structure (step 6). |
| `GGML_OPENVINO_FORCE_F32=1` | Pin the plugin to f32 ACCURACY mode to test bf16 rounding accumulation. |
| `GGML_OPENVINO_PROFILING=1` | Per-stage timing (decode/convert/compile/infer). |
| `GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS=1` | Tensor address map (trace pointer aliasing between subgraphs). |

The frontend converter carries no env-var debug seams — keep them in the backend. When an op has
two conversion strategies, expose the alternate as an op attribute (e.g. GatedDeltaNet's
`force_ref` selects the serializable Loop path) so a unit test can pick it via `SingleOpBuilder`
without touching process env or the production default.

---

## Gotchas

- **Big models OOM in the frontend, not just in weight RAM.** Quant types in `supported_types`
  are materialized as OV constants *in addition to* ggml's copy, ~doubling peak compile memory.
  Reducing `-c` doesn't help (the blow-up is per-subgraph compilation, not the KV cache). A Q4_K
  model can OOM where the same-arch Q2_K runs (Q2_K isn't in `supported_types`, stays on CPU).
  Exit 137 = OOM-killed.
- **Iterate on a small quant.** BF16 of a 4B model can take ~25 min to load/compile; Q4_K_M loads
  in seconds and reproduces the same *structural* bugs. Reach for higher precision only if step 2
  implicates quantization.
- **Per-subgraph vs shared runtime state.** A model that splits into many subgraphs per token step
  must not share single-value bookkeeping across them (e.g. `stateful_kv_size` belongs on the
  per-subgraph context, not the shared one) — a shared counter advances once per subgraph and
  mis-slices state. Symptom: a ROI error at the *second* decode step, not the first.
- **An old reference build can predate the arch** and segfault on load — then it can't be the
  reference; use a fresh `-DGGML_OPENVINO=OFF` build at the current commit (step 1).
- **Shell:** piped `llama-*` output buffers and never flushes — redirect to a file and grep it.
  `cd` doesn't persist between calls (use absolute paths). `pkill -9 -f llama` between runs so
  stale processes don't starve cores.

---

## Checklist

1. **Is it an OV bug?** `build-cpu` (`-DGGML_OPENVINO=OFF`) vs `build-ov`, greedy, on
   `llama-simple`. CPU good + OV bad = OV bug. Both bad = shared (build/tokenizer/template), stop
   debugging OV.
2. **Graph or quant?** Same weights, CPU ref vs OV. Wrong on OV only → graph bug, not quant.
3. **Prefill or decode?** `-n 1` vs `-n 12`.
4. **First-divergence diff** (`llama-eval-callback`, positional `sum` diff) → first node past
   rounding. Highest-yield step.
5. **Per-layer sweep with element-wise cosine**, never Min/Max/Mean. Cliff = localized structural
   bug; smooth slope = drift. Never conclude "drift" from a permutation-invariant metric.
6. **`DISABLE_TYPES=<all quant>`** → still wrong ⇒ structural (f32) bug, quant ruled out.
7. **`DISABLE_OPS`** to force a suspect op to CPU; binary-search. Correct = culprit; new crash =
   inconclusive (re-split artifact).
8. **Layout pinpoint:** full-element dumps → multiset compare. Sorted-match high + cosine low =
   layout/wrong-source bug; match the distribution against candidate references to name the
   source. For a `VIEW`, inspect `nb[]` + `op_case`, not just `ne[]`.
9. **Op-level test** in `test_ops.cpp` at real dimensions (H, GQA, T, B not collapsed), against a
   **ggml-CPU oracle**, never a hand-derived reference; assert every conversion path; leave it as
   a regression guard.
10. **Cross-arch regression:** after any shared-path fix, re-run the other supported models.

> Only a divergence you can *eliminate* — and thereby fix the output — is the real culprit; the
> first visible one is often benign rounding.
