# Debugging Accuracy Issues in the GGUF Frontend / OpenVINO Backend

How to find *why* a GGUF model produces wrong output when run through the OpenVINO
frontend + `ggml-openvino` backend, as opposed to the reference llama.cpp CPU path.

This is a field guide, not a spec. It captures the techniques that have actually
paid off (and the traps that wasted time) while bringing up qwen3-next and other
architectures. Read it before reaching for a debugger.

---

## The one rule: always have an authoritative reference

Every accuracy claim is a comparison. The single most important decision is *what
you compare against*, and the answer is almost always **the real llama.cpp CPU
implementation** — never a reference you derived by hand.

A hand-written reference (a numpy script, a C++ recurrence you wrote from the paper)
encodes *your* understanding of the op's layout and math. If that understanding is
wrong in the same way the frontend is wrong, the two agree and the bug hides. Worse:
if your hand-reference is wrong but the frontend is right, you'll "find" a bug that
doesn't exist and waste a day.

Concrete example from qwen3-next: a hand-written GatedDeltaNet reference packed the
final recurrent state as `[value][head][key]`; ggml's own kernel packs it
`[head][value][key]`. With one head (`H=1`) the two orders are identical, so the
existing single-head unit test passed and gave false confidence. Only when the
reference was regenerated from **ggml's own CPU kernel** did the true layout appear —
and it revealed the frontend was actually *correct*, sending the investigation
elsewhere.

> If you find yourself typing out the math of an op to build an expectation, stop and
> generate the expectation from ggml instead (see "The ggml-CPU oracle" below).

---

## Bisection strategy, coarse to fine

Work from the cheapest, coarsest signal to the most expensive, finest one. Don't open
a debugger until the coarse steps have cornered the bug.

### 1. Confirm it's a graph bug, not quantization

Run the **same quantized weights** through the reference CPU path and through the OV
backend, both greedy (`--temp 0 --top-k 1`), and compare the first generated token:

```
# reference (plain CPU llama.cpp build)
./build-ref/bin/llama-completion -m model.gguf -p "The capital of France is" \
    -n 1 -no-cnv --no-warmup --temp 0 --top-k 1

# OpenVINO backend
GGML_OPENVINO_DEVICE=CPU ./build-ov/bin/llama-completion -m model.gguf \
    -p "The capital of France is" -n 1 -no-cnv --no-warmup --temp 0 --top-k 1
```

If the reference is correct on the same weights and OV is wrong, quantization is
**ruled out** — it's a graph/conversion bug. Don't go download a higher-precision
model to "check if quantization is the problem"; the same-weights comparison already
answered that, for free. (Higher-precision models are also often too large to
materialize through the frontend — see "Gotchas".)

### 2. Localize to prefill vs decode

Run with `-n 1` (prefill only) vs `-n 12` (prefill + decode). If the **first** token
is already wrong, the bug is in the prefill graph and you can ignore all
state/KV-cache/decode machinery. If prefill is right and later tokens drift, suspect
stateful bookkeeping, the KV/recurrent-state round-trip, or a decode-only code path.

### 3. Localize to an op family by forcing ops onto the CPU reference

This is the highest-leverage backend technique and the reason the backend has a
`supports_op` gate. `ggml_backend_openvino_device_supports_op()` in
`ggml/src/ggml-openvino/ggml-openvino.cpp` decides, per op, whether OV claims it or it
falls back to ggml's CPU kernel. **Temporarily returning `false` for a suspect op
forces ggml's reference implementation for that op only**, while everything else stays
on OV. If accuracy is restored, the bug is in OV's handling of that op (or how it's
fed); if not, the op is exonerated.

Add a temporary env-gated escape at the top of `supports_op`:

```cpp
// DEBUG bisection: force a suspect op onto ggml CPU. Remove once diagnosed.
if (getenv("GGML_OPENVINO_NO_SSM_CONV") && op->op == GGML_OP_SSM_CONV) {
    return false;
}
```

Rebuild the backend, then toggle with the env var — no source edit per experiment:

```
GGML_OPENVINO_NO_SSM_CONV=1 GGML_OPENVINO_DEVICE=CPU ./build-ov/bin/llama-completion ...
```

Binary-search the op set: disable half the suspect ops, see if it fixes the output,
then narrow. For a novel architecture, start by disabling the arch-specific ops (the
ones only that model uses) since the shared ops are already exercised by working
models.

**Important caveat — read the result carefully.** Moving an op to CPU changes where the
ggml scheduler splits the graph into OV subgraphs. That re-split can surface a
*different, unrelated* error (typically an OV shape-mismatch at a new subgraph
boundary) that masks the accuracy signal you were after. So:

- **Output becomes correct** → strong signal: that op (or its glue) was the bug.
- **Output still wrong, same as before** → strong signal: op exonerated.
- **A new crash / shape error appears** → *inconclusive*, not a confirmation of guilt.
  The re-split hit a boundary bug. Note it, but don't read it as "this op is broken."
  Try disabling a different combination, or move to the op-level oracle test (below).

### 3b. First-divergence diff with the eval-callback (the highest-yield method)

Op-forcing tells you *whether* an op matters; it does not tell you *where* the graph
first goes wrong, and (as the caveat above shows) it frequently self-destructs on a
re-split. The complementary technique — and in practice the fastest way to localize —
is to dump **every node's output on both backends and diff them in graph order**.

`llama-eval-callback` (in both `build-ref/bin` and `build-ov/bin`) fires
`common_debug_cb_eval` per ggml node and prints the node name, op, source shapes, and
a per-tensor **`sum`**. That `sum` is a cheap, position-stable fingerprint. Node names
are identical across backends, so aligning the two logs **positionally** (not by name —
names like `norm-0`, `ffn_moe_weighted-0` repeat across layers and experts, so a
name-keyed join silently mismatches) surfaces the first node whose sum differs by more
than f32 rounding.

```
# Reference (ggml CPU). NOTE: llama-eval-callback rejects --no-warmup; just omit it.
./build-ref/bin/llama-eval-callback -m model.gguf -p "The capital of France is" -n 1 > /tmp/cb_ref.log 2>&1

# OpenVINO
GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=1 \
    ./build-ov/bin/llama-eval-callback -m model.gguf -p "The capital of France is" -n 1 > /tmp/cb_ov.log 2>&1

# name<TAB>sum in emission order, then align positionally and flag >0.1% diffs
awk '/cb_eval:/{n=$2} /sum = /{print n"\t"$3}' /tmp/cb_ref.log > /tmp/ref_pos.txt
awk '/cb_eval:/{n=$2} /sum = /{print n"\t"$3}' /tmp/cb_ov.log  > /tmp/ov_pos.txt
paste /tmp/ov_pos.txt /tmp/ref_pos.txt | awk -F'\t' '{
  d=$2-$4; if(d<0)d=-d; b=($4<0?-$4:$4); if(b<1)b=1; rel=d/b;
  flag=($1!=$3)?" NAME-MISMATCH":(rel>0.001?" <<< DIFF":"");
  printf "%3d %-26s ov=%-12s ref=%-12s%s\n", NR,$1,$2,$4,flag }'
```

Reading the result — the important judgement calls, all learned the hard way:

- **The first *visible* divergence is usually NOT the root cause.** On qwen3-next the
  first `<<< DIFF` was a tiny (0.12%) `ssm_ba` matmul; forcing its weight type to CPU
  (step 3c) did *not* fix the output, so it was benign kernel rounding. Confirm any
  candidate by actually eliminating it, don't assume.
- **Ignore intermediate nodes that live *inside* an OV subgraph.** With the OV backend,
  only nodes at subgraph split boundaries carry real data back to ggml; interior nodes
  (often the `(view)` / `(reshaped)` / `cache_*` entries) read as `0` or stale. A
  zero-vs-nonzero "difference" on those is an artifact, not a bug.
- **The eval-callback perturbs OV's graph splitting** (it forces many nodes to be
  boundaries) and can crash the OV run early with `map::at` or a vector-size error.
  You still get the prefix up to the crash, which is usually enough to localize the
  *first* divergence. For a full-depth sweep that does **not** perturb the split, use
  `GGML_OPENVINO_DEBUG_OUTPUT` instead (below).
- **Positional alignment only holds until the two backends' node *order* diverges.**
  The OV backend may emit its nodes in a different order (or a different count — e.g.
  extra state-copy nodes) than ggml. Once that happens the `paste`-based alignment
  shifts by one and every subsequent row shows a spurious `NAME-MISMATCH`. Trust rows
  up to the first genuine name mismatch; beyond it, filter to rows where the two names
  still agree (those remain valid same-tensor comparisons) rather than reading the
  shifted values.

### 3c. Full-depth sweep with GGML_OPENVINO_DEBUG_OUTPUT

`GGML_OPENVINO_DEBUG_OUTPUT=1` prints First/Min/Max/**Mean** for every *subgraph
output* at its natural boundary — so it neither perturbs the split nor crashes, and it
covers all layers to the final `result_output`. Pick a full-width, same-named,
non-normalized tensor that exists in both logs (e.g. `attn_residual-N`, the post-
attention residual per layer) and plot its statistic across all N layers against the
reference. This answers "does the error appear suddenly at one layer (structural bug)
or accumulate gradually (many-small-op drift)?" — a distinction op-forcing can't make.
Prefer Min/Max or a wide-dynamic-range value over Mean for normalized tensors, whose
mean sits near zero and makes relative diff meaningless noise.

**Two traps that faked a "sharp layer-0 bug" on qwen3.5 — both cost real time:**
1. *Trust only state-checkpoint outputs, never intermediate ggml-node sums under the
   eval-callback.* Because the callback perturbs OV's split (§3b), OV materializes stale
   buffers for interior nodes — a **pure `TRANSPOSE` appeared to change its element sum**,
   which is mathematically impossible. That is diagnostic *of the perturbation*, not of a
   transpose bug. The state writes (`cache_s_lN` recurrent state, `cache_r_lN` conv state)
   and `result_output` are genuine subgraph boundaries and are the *only* interior values
   safe to compare cross-backend.
2. *Parse the fixed-width First/Min/Max/Mean dump as four proper floats.* The columns run
   together with no guaranteed space (e.g. `0.48834-4.64966e-05` = Max `0.48834`, Mean
   `-4.64966e-05`). A naive "last number on the line" or greedy-digit regex glues the
   negative Mean's exponent onto Max and invents absurd values — this faked a 7000× state
   "explosion" at the last two layers that did not exist (real means were all ~1e-4 with
   sane min/max). Split with a float regex anchored after the `Mean` header and sanity-check
   against Min/Max before believing any large ratio.

**Cross-arch confirmation.** Running this on qwen3.5-4B (dense, `general.architecture=qwen35`)
gave what *looked* like the same verdict as qwen3-next: layer-0 recurrent state matches the
reference (ratio 0.99), then the per-layer `cache_s` ratio appears to wander diffusely
(0.15–2.65) with no single broken layer and no blow-up. Because qwen3.5-4B is **dense**, this
also exonerates MoE routing as the shared cause, and a 4B model reproduces it in seconds — use
it, not the 80B, as the iteration target.

> **⚠️ The "diffuse drift" verdict for qwen3.5 was WRONG — and the reason it was wrong is the
> single most important lesson in this guide.** The Min/Max/Mean statistics `DEBUG_OUTPUT`
> prints are **permutation-invariant**: they do not change when the *elements* of a tensor are
> reordered, only when their *values* change. A layout/head-scramble/partial-rotary bug moves
> correct numbers to wrong positions — the per-layer mean barely moves, so the ratio "wanders"
> in a way that is indistinguishable from genuine precision drift. It is not drift; it is a
> structural bug hiding behind a permutation-blind metric. See "The permutation-invariance trap"
> below. On qwen3.5 the real cause was a single localized op bug (partial-rotary IMROPE), found
> only after switching to an element-wise cosine metric.

**The full-depth cross-backend diff (the one that actually settles structural-vs-drift).**
The eval-callback (§3b) sees both backends but OV crashes early on the perturbed split,
so it only covers the first layer or two. `DEBUG_OUTPUT` reaches all layers but is
OV-only. Combine them: take the **complete** reference `llama-eval-callback` log (ggml
CPU runs to the end, hundreds of thousands of lines) and match its per-layer `l_out-N`
(or `attn_residual-N`) `sum` — divided by element count to get a mean — against the OV
`DEBUG_OUTPUT` mean for the *same* layer. Now you have both backends at **every** layer.
Filter the reference log to the real prefill batch by shape (e.g. ggml `{2048,5,1,1}`
for T=5), since warmup/decode batches reuse the same node names. On qwen3-next this
showed the per-layer mean diff sitting at rounding through ~layer 6, first exceeding it
at layer 7, then climbing **monotonically** to the last layer with OV *consistently*
more negative than ref — a systematic directional drift, not symmetric noise. That
pattern (small, one-signed, compounding, no single jump) is the fingerprint of a mildly
biased op applied on *every* layer, and it rules out the "one broken layer" hypothesis
that op-forcing and single-layer eval-callback diffs keep tempting you toward.

### 3c-bis. Logit-margin probe (is the model actually far from right, or just tipped?)

Before hunting a "bug," measure how badly wrong the output is. Bias the reference toward
its correct token with `-l <TOKEN_ID>-N` (subtract N from that logit) and the OV run
toward *its* wrong token with `-l <TOKEN_ID>+N`, bisecting N until the top-1 flips. On
qwen3-next the correct "Paris" beat its runner-up by only ~4 logits in the reference,
while OV's chosen token sat ~7-8 logits / ~7th place behind — so OV is not merely
losing a photo-finish, it promotes a far-down token, i.e. a real distortion rather than
a tie broken by rounding. A *tight* reference margin plus a *large* OV displacement is
the profile of "small systematic per-layer drift amplified by a knife-edge final
softmax," which points at accumulated precision loss rather than one catastrophic op.

### 3c-ter. The permutation-invariance trap — use element-wise cosine, not Min/Max/Mean

This is the lesson qwen3.5 taught the hard way, and it retroactively invalidates the
"diffuse drift" reading above. **Min, Max, Mean, and `sum` are all invariant under
permutation of a tensor's elements.** A whole class of bugs only *reorders* correct values:

- partial-rotary rope that rotates the wrong slice of a head (values are right, positions wrong),
- a head-scramble from a bad reshape/transpose (`[H,D]` read as `[D,H]`),
- an interleaved-vs-split-halves mismatch,
- any layout error on an axis a size-1 dim would have collapsed.

For every one of these, the per-layer mean/min/max is essentially unchanged, so a
statistic-based sweep reports "small diffuse drift, no single broken layer" — the exact false
verdict that sent qwen3.5 chasing precision for a day. **A permutation-invariant statistic
cannot see a permutation bug. Period.**

The metric that *can* see it is **element-wise cosine similarity** between the OV tensor and
the reference tensor, computed position-by-position on the flattened buffers:

```
cos = dot(ov, ref) / (norm(ov) * norm(ref))
```

Cosine is ~1.0 when the two agree elementwise and collapses the instant elements are
reordered. Sweep it per layer (same tensor-picking rules as §3c: full-width, same-named,
genuine subgraph boundaries — `cache_s_lN`, `cache_r_lN`, `attn_residual-N`, `result_output`).
The shape of the cosine curve *is* the diagnosis:

- **A sharp cliff** at one layer (e.g. 0.999997 → 0.43 between adjacent layers) = a **localized
  structural bug** at that layer. This is what a real bug looks like.
- **A gentle, monotone slope** from ~1.0 downward = genuine accumulating precision drift.

On qwen3.5 the cosine sweep showed exactly a cliff: GDN layers 0–2 (which carry **no rope**)
held cos 0.999997; layer 4 — the first GDN layer *after* full-attention layer 3 applied rope —
dropped to cos 0.43. That single discontinuity, aligned to "the first tensor downstream of a
rope layer," pointed straight at rope. It was then confirmed structurally from the GGUF
metadata: `<arch>.attention.key_length = 256` but `<arch>.rope.dimension_count = 64`, i.e.
**partial rotary** (only the first 64 of each 256-dim head rotate). The fix is in
`src/frontends/gguf/src/op/rope.cpp` (see "Bug archetypes" below).

> Rule of thumb: the moment a statistic-based sweep says "diffuse drift, no clear culprit,"
> do **not** believe it — re-run with element-wise cosine before concluding drift. Drift is a
> conclusion you earn only when cosine *also* shows a smooth slope, never when a
> permutation-blind metric fails to find a jump.

### 3c-quater. The decisive OV-bug-vs-shared-bug test: a SEPARATE CPU-only build

Once the graph runs end-to-end but the *output text* is bad, the first question is no longer
"which op" but "is this even an OV bug at all?" You answer it by comparing OV against a genuine
ggml-CPU run of the **same model at the same commit**. The trap is getting a real CPU run.

> **⚠️ THE `build-ov` BINARY HAS NO CPU BACKEND. Every "CPU" knob you reach for silently runs
> OpenVINO.** When llama.cpp is built with `-DGGML_OPENVINO=ON`, the OV backend registers as a
> ggml *device* (`OPENVINO0`) and there is **no CPU device registered at all**
> (`llama-cli --list-devices` shows only `OPENVINO0`). Consequences, each of which faked a "CPU
> reference" and cost real time:
> - `env -u GGML_OPENVINO ...` does **nothing** to the backend — it is a *device*, not gated by
>   that env var. (`GGML_OPENVINO=""` likewise still runs OV.)
> - `--device none` still runs OV (`OpenVINO: using device CPU` in the log is OV's *internal*
>   device, not ggml's CPU backend).
> - `GGML_OPENVINO_DEVICE=CPU` runs OV-on-CPU — still the OV graph, not ggml.
>
> So "I compared CPU vs OV and they matched → shared bug" was actually **OV vs OV**. On qwen3.5
> this produced a totally wrong "the port is faithful, my work is done" conclusion. The output
> was degenerate on OV and *correct on a real CPU build*.

The **only** reliable CPU reference is a second build with the OV backend compiled out, at the
same commit:

```
cmake -B build-cpu -DGGML_OPENVINO=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build-cpu --target llama-simple llama-eval-callback -j$(nproc)
```

Then compare, greedy, on `llama-simple` (raw completion — **no chat template**, see gotcha):

```
./build-cpu/bin/llama-simple -m model.gguf -n 20 "The capital of France is"   # true ggml CPU
./build-ov/bin/llama-simple  -m model.gguf -n 20 "The capital of France is"   # OpenVINO
```

- **Both good** → port correct, done.
- **CPU good, OV bad** → a genuine OV bug (qwen3.5: CPU → "Paris.", OV → garbage). Localize with
  §3b–§3d against the `build-cpu` eval-callback.
- **Both bad** → only *now* may it be a shared llama.cpp/model/template issue — but confirm the
  CPU build is genuinely OV-free (`--list-devices`) before believing it.

> **Gotcha — the chat template hides the transformer signal.** `llama-cli` (and even
> `llama-completion`) apply the model's chat template; a thinking model like qwen3.5 (Qwen3-VL
> template) then emits `<think>` (printed as `[Start thinking]`) and reasons instead of
> completing, so greedy output looks degenerate even when the math is fine. Use **`llama-simple`**
> for accuracy bisection — it does raw greedy completion with no template. Reserve `llama-cli`
> for judging end-user quality *after* the math is verified.
> Also: `cd` does not persist between separate shell invocations here, and piping `llama-*`
> through `tail`/`grep` buffers and never flushes — redirect to a file (`> log 2>&1`) and grep
> the file; `pkill -9 -f llama` between runs so stale processes don't starve cores.

### 3d. Isolate a suspect quant path with GGML_OPENVINO_DISABLE_TYPES

Forcing an *op* to CPU re-splits the graph and often crashes. Forcing a *type* to CPU
is far more surgical: it moves only the tensors of that quant type onto ggml while
leaving the graph structure intact. `GGML_OPENVINO_DISABLE_TYPES=Q8_0,Q6_K` removes
those from `supported_types` for the run (via the same debug seam as
`GGML_OPENVINO_DISABLE_OPS`). Two high-value moves:

- Disable **one** suspect type to test whether *that* quant kernel is the culprit.
- Disable **all** materialized quant types at once
  (`Q4_0,Q4_1,Q4_K,Q5_K,Q8_0,Q6_K,Q5_1,MXFP4`). Now every weight matmul runs on ggml
  with bit-identical weights to the reference, and OV executes only the f32/f16
  structural ops (norm, rope, attention, GDN, ssm_conv, softmax, elementwise glue). If
  the output is *still* wrong, you have **decisively ruled out every quant kernel** and
  proven the bug is structural — an enormous narrowing that no single op-force gives
  you. (This is exactly how qwen3-next's bug was confirmed to be in the f32 path, not
  quantization.)

### 3e. Pinpoint a layout / wrong-source bug: full-element dumps, multiset, distribution matching

Cosine (§3c-ter) tells you *where* the graph first scrambles and *that* the bug is a
permutation, not arithmetic. It does not tell you **what** the wrong tensor is a permutation
*of* — and that is usually the whole answer. A scramble almost always means the op read the
**wrong source region** (a neighbouring head, the sibling half of a joint projection, a
strided window mistaken for a contiguous one). This step names that source.

**Dump full element order, not just statistics.** Min/Max/Mean/sum are permutation-invariant
(§3c-ter) and useless here. Use the element-order-preserving dump seams — one per backend —
so you can compare buffers position-by-position *and* as sorted multisets:

- OV side: `GGML_OPENVINO_DUMP_TENSOR=<name-substr>` writes each matching model-output tensor
  to `/tmp/ov_dump_<name>_<n>.txt`, one float per line, in element order.
- CPU reference side: `GGML_DUMP_TENSOR=<name-substr>` (the ggml eval-callback path) writes the
  same for the ggml-CPU run to `/tmp/ref_dump_*` / its logged path.

Run both on the same prompt and the same layer's tensor, then apply two comparisons in order:

1. **Multiset (sorted) comparison — the layout-vs-math discriminator.** Sort the nonzero
   values of both dumps and compare the sorted sequences (a sorted cosine or elementwise
   diff of the two sorted vectors). This is invariant to *any* permutation:
   - Sorted match ≈ 1.0 but position-wise cosine low ⇒ **the numbers are all correct, only
     their positions are wrong** — a pure layout/read bug (reshape, transpose, strided view,
     interleave). Now you know to look at *layout*, not at the math of any kernel.
   - Sorted match also low ⇒ the values themselves differ — a real arithmetic bug (wrong
     scale, wrong op, precision). Go back to op-forcing / oracle tests.

2. **Distribution matching against *multiple* candidate references — the wrong-source finder.**
   When step 1 says "right numbers, wrong place," the values had to come from *somewhere*.
   Dump several *candidate* reference tensors that live near the suspect in the graph (the
   sibling projection half, the adjacent head block, the pre-/post-transform version) and
   compare the OV tensor's **distribution** (sorted multiset, plus range/mean/std as a quick
   fingerprint) against each. The OV tensor will match the distribution of the tensor it
   **actually read**, which is frequently *not* the one it was supposed to. "OV's X matches
   the reference's Y, not the reference's X" localizes the bug to the op that produced X and
   tells you exactly which region it grabbed instead.

**The trustworthiness caveat that makes or breaks this step.** Interior tensors are **not**
generally safe to dump: ggml reuses scratch buffers, so promoting an arbitrary interior node
to a model output can read memory that was already clobbered by a later op — you get
plausible-looking garbage that is neither the value you wanted nor a stable reference. Rules
that hold up:

- **Only persistent buffers give trustworthy interior comparisons** — the KV/recurrent-state
  caches (`cache_k`, `cache_v`, `cache_r`, `cache_s`, …) and genuine subgraph-boundary
  outputs. These are not scratch, so their contents survive to the dump. Confirming the cache
  writes match the reference (they usually do) also *bounds* the bug: everything feeding the
  cache is correct, so the fault is downstream of it.
- **Promote at most one interior tensor at a time.** A single-tensor ("alone") promotion is
  reliable enough for distribution/multiset checks; promoting several interior tensors
  simultaneously perturbs allocation and corrupts them all. If you need N interior tensors,
  do N separate runs.
- A post-fix dump of an interior tensor that still looks "wrong" is therefore **not** proof
  the fix failed — it may be the aliasing artifact. The authoritative post-fix check is the
  end-to-end output text (§3c-quater), never an interior promotion.

**Then read the offending op's layout directly.** Once the wrong-source op is named — very
often a `VIEW`/reshape — dump its actual ggml geometry (`ne[]`, `nb[]` strides, offset) and
its backend *classification* (the `op_case` the decoder assigned it). A view whose source is
strided or gapped (head stride ≠ head extent, a sub-block selected out of a larger group) but
which was classified as a *dense/contiguous* window is the bug: the contiguous handler slices
a run of elements that straddles the gap and scrambles them. Fix the classification (a
density/stride check that rejects the strided case and routes it to a stride-aware handler),
not the downstream consumer. `GGML_OPENVINO_DUMP_CGRAPH=1` gives the `ne[]`; a temporary
`getenv`-gated `fprintf` of `nb[]`/offset/op_case at the classification site gives the rest.

> Whenever you fix a VIEW/reshape classification, re-verify it does **not** reroute a *different*
> architecture's views. A stride/density predicate that newly rejects arch A's view can also
> reject arch B's legitimately-contiguous view. Re-run the other supported models and confirm
> their view classification is unchanged (log the `op_case` for a known view and diff it) — see
> the cross-architecture regression check in the checklist.

### 4. Reproduce at the op level with a ggml oracle test

Once an op is implicated, stop testing through the 27 GB model and reproduce it in the
frontend's own unit tests (`src/frontends/gguf/tests/test_ops.cpp`), which build a
one-op `ov::Model` via `SingleOpBuilder` and run it on CPU in milliseconds. This is
where the bug gets pinned and where the regression test lives afterward.

The key is to feed the op **dimensions that match the real model**, not just the
minimal `1×1×1` case. Most layout bugs live on an axis that a size-1 dimension
collapses:

- multi-head packing (`H>1`) — collapsed by `H=1`
- GQA head repeat (`H_v != H_k`) — collapsed by equal head counts
- multi-token (`T>1`) — collapsed by `T=1`
- batch (`B>1`) — collapsed by `B=1`

Pull the real op's input shapes from a cgraph dump (`GGML_OPENVINO_DUMP_CGRAPH=1`,
see below) so the test exercises the same axes the model does. A test that passes at
`H=1` but the model uses `H=32` has proven nothing about the failing case.

---

## The ggml-CPU oracle

To get ground-truth values for an op-level test, build a tiny standalone C program
that links llama.cpp's `libggml` + `libggml-cpu`, constructs a one-node graph for the
op, runs it on the CPU backend, and prints the output. Those printed numbers become
the `expected` array in the unit test. `src/frontends/gguf/tests/gdn_oracle.c` is a
worked example for `GGML_OP_GATED_DELTA_NET`; copy its structure.

Build/run recipe (adjust the op and shapes):

```
gcc gdn_oracle.c -o gdn_oracle \
    -I <llama.cpp>/ggml/include \
    -L <llama.cpp>/build-ov/bin -lggml -lggml-base -lggml-cpu -lm \
    -Wl,-rpath,<llama.cpp>/build-ov/bin
LD_LIBRARY_PATH=<llama.cpp>/build-ov/bin ./gdn_oracle
```

Oracle gotchas learned the hard way:

- **Link against the *same* libggml the backend uses** (here, `build-ov/bin`). A
  different build tree can be stale and assert on a *different* tensor shape than the
  source you're reading — you'll chase a mismatch that only exists in the old binary.
- Set `no_alloc = true` in `ggml_init_params` when you use
  `ggml_backend_alloc_ctx_tensors`; otherwise the allocator asserts.
- Read the op's shape contract from the **header** (`ggml.h` documents q/k/v/g/beta/
  state layouts for GDN, for instance) rather than guessing `ne[]` order.
- Fill inputs with distinct, asymmetric values (not all-ones, not `0,1,2,3`
  symmetric) so a transposed or mis-strided axis actually changes a number.

Once you have oracle values, assert **both** conversion paths against them if the op
has more than one (e.g. GDN's fused `ov::op::internal` op *and* its decomposed Loop
fallback). Divergent paths that give *different* wrong answers tell you at least one
mishandles an axis; matching a shared oracle tells you both are correct there.

---

## Bug archetypes seen in practice

The concrete structural bugs found so far cluster into three shapes. When a cosine cliff or a
dynamic-shape crash points at one of these areas, check the archetype first.

### A. Partial-rotary rope (a config field ignored by the op)

qwen3.5 full-attention layers have `head_dim = 256` but `rope.dimension_count = 64`: only the
first 64 dims of each head rotate, the tail passes through unchanged. The IMROPE (and NEOX)
translator in `op/rope.cpp` originally took the rotary width from `output_shape[3]` (=256, the
whole head) and rotated everything, corrupting the pass-through tail on **every**
full-attention layer. The fix mirrors the NEOX partial handling: take
`n_rot = rope_config.n_dims > 0 ? rope_config.n_dims : head_dim`, `Slice` the head into
`rotary_in` (`0..n_rot`) + `pass_through` (`n_rot..head_dim`), rotate only `rotary_in` with
cos/sin of width `n_rot/2`, then `Concat([rotated, pass_through], axis 3)`.

Lesson: **any rope/attention op must read its geometry from the model config, not from the
tensor's last dim.** The last dim is the *full* head; partial rotary makes the two differ. A
unit test at `head_dim == n_dims` (full rotary) cannot catch this — the guard test
`ImropePartialRotary` deliberately uses `head_dim=256, n_dims=64`.

### B. Shared precomputed tables built with the wrong config flag

The sin/cos table is precomputed **once** and shared across all rope layers
(`translate_session.cpp::add_rope_sin_cos` → `make_sin_cos`). For interleaved M-RoPE (qwen3.5 /
qwen3vl) the position input carries **4 mrope sections** (`4 × n_tokens` ids), so the table's
token axis is 4× a NEOX model's. `make_sin_cos` was called with `imrope` defaulting to
`false`, so the shared table came out with the wrong token dimension and later broadcast-crashed
at the rope `Multiply` (`Multiply_* eltwise shape mismatch at dim 1`) once archetype A exposed
it. Fix: thread the real flag — `RopeConfig::is_imrope` (set in `ggml-decoder.cpp
get_rope_config` from `op_params[2] == GGML_ROPE_TYPE_IMROPE`) → passed into `make_sin_cos`.

Lesson: **a table shared across layers must be built with the same per-op config the consuming
op assumes.** When a shared/precomputed value and its consumer disagree on a config flag, the
symptom is a broadcast/eltwise shape mismatch, not a wrong number — and the old code can *mask*
it (the buggy IMROPE path silently reshaped cos to a self-consistent-but-garbage width).

### C. Dynamic-axis tail slices that bake an absolute offset

`conv_state_last` reads the last `d_conv-1` (=3) columns of the conv window, whose length
`ncs = n_tokens + d_conv - 1` is **dynamic**. The decoder emitted the slice with an **absolute**
start computed at prefill (`prefill_ncs - 3 = 2`); at decode `ncs` shrinks to 4, so `Slice[2,5)`
clamped to width 2 and the downstream reshape `(1,1,8192,2) → (1,1,1,24576)` failed. Fix:
**end-anchor the tail** — when a slice is a tail (`start + len == axis_length`) emit a
**negative** start (`start = -len`) in `ggml-decoder.cpp`, and in `op/view.cpp` op_case 3 a
negative start ⇒ `end = INT64_MAX`. Same negative-index idiom as the GDN op_case 4 state slice.

Lesson: **any slice on a dynamic axis must be anchored to the end it's actually pinned to.** A
tail is pinned to the axis end, so index it from the end (negative start, open end), never from
an offset baked at prefill — that offset is only correct for the prefill length.

> All three share a meta-pattern: a value that is *correct for prefill* (full head, NEOX token
> count, prefill window length) silently becomes wrong at decode or for a different arch,
> because the code hard-coded the prefill/common-case shape instead of reading the true
> per-op/per-step geometry.

### D. A strided VIEW mis-classified as a contiguous window (heads scrambled)

The backend classifies every ggml `VIEW` into an `op_case` (contiguous shrink, feature-window,
packed-output, …) that picks the frontend slicing strategy. When a model packs two logically
separate things **interleaved** on one axis — e.g. a joint projection laid out per head as
`[A_h0, B_h0, A_h1, B_h1, …]`, where a view for `A` selects one sub-block out of each stride —
the view's ggml strides show the tell: the group stride is larger than the extent the view
keeps (a gap between the blocks it reads). If the classifier treats that view as a *dense*
feature-window, the contiguous handler slices `[off : off+len]` straight through the gap,
mixing `B` into `A` and shifting every head. The K/V paths, which use plain (non-interleaved)
reshapes, stay correct — so caches match the reference while the interleaved projection is
scrambled, exactly the "right numbers, wrong place" signature of §3e.

The fix is two-part: (1) a **density/stride guard** on the contiguous-window detector so it
rejects any view whose feature dims (sorted by stride) do not tile without a gap — this stops
the strided view from hijacking the contiguous `op_case`; and (2) a **stride-aware handler**
(a new `op_case`) that reshapes the source to expose the group stride, `Slice`s the intended
inner sub-block on the stride axis, then reshapes to the view's own layout. Because the
detector is consulted *before* the fallback, the guard in (1) is the load-bearing edit — without
it the strided view keeps matching the contiguous case and (2) never fires.

Lesson: **a VIEW's `op_case` must be decided from its actual strides, not from its element
count.** Two views with identical shapes and offsets — one contiguous, one interleaved with a
gap — need different handlers; only `nb[]` distinguishes them. And any predicate change here is
cross-arch load-bearing (see the regression note in §3e).

---

## Backend debug env vars

All defined in `ggml/src/ggml-openvino/`. Gate any new debug output on a `getenv`
check and write to `stderr` — `GGML_LOG_INFO` is suppressed in some builds.

| Env var | Effect |
|---|---|
| `GGML_OPENVINO_DEVICE=CPU` | Run OV on CPU (comparable to ggml CPU). |
| `GGML_OPENVINO_DUMP_CGRAPH=1` | Dump each OV subgraph as `cgraph_ov_N.txt` — node ops, names, and **per-tensor `ne[]` shapes**. The primary tool for reading the real op dimensions and how the graph was split. |
| `GGML_OPENVINO_DUMP_IR=1` | Serialize each compiled subgraph to `model_*.xml` (inspect the actual OV graph, dynamic dims, etc.). Not available for models containing internal ops (non-serializable). |
| `GGML_OPENVINO_DEBUG_INPUT=1` | Print each bound input tensor's name/shape/values before infer. |
| `GGML_OPENVINO_DEBUG_OUTPUT=1` | Print each output tensor after infer. |
| `GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS=1` | Print tensor address map (trace pointer-based aliasing between subgraphs). |
| `GGML_OPENVINO_STATEFUL_EXECUTION=1` | Enable the stateful KV/recurrent-state path (needed for the real decode flow). |
| `GGML_OPENVINO_PROFILING=1` | Per-stage timing (decode/convert/compile/infer). |
| `GGML_OPENVINO_DISABLE_OPS=SSM_CONV,DIV` | Debug seam: force listed ggml ops onto the ggml CPU reference (comma-separated ggml op names). Re-splits the graph — a new crash is inconclusive (see step 3). |
| `GGML_OPENVINO_DISABLE_TYPES=Q8_0,Q6_K` | Debug seam: force tensors of the listed ggml quant types onto ggml CPU without changing graph structure (see step 3d). |
| `GGML_OPENVINO_DUMP_TENSOR=<name-substr>` | Dump the **full element-order** contents of each matching OV output tensor to `/tmp/ov_dump_*` (one float/line). For the position-wise + multiset comparison of §3e; unlike `DEBUG_OUTPUT` it is not permutation-invariant. |
| `GGML_DUMP_TENSOR=<name-substr>` (ggml `common/debug.cpp`) | The CPU-reference counterpart: dumps matching tensors from the ggml eval-callback run in element order. Diff against the OV dump. |

Frontend-side seams for A/B testing without a backend rebuild:

| Env var | Effect |
|---|---|
| `GGUF_GDN_FORCE_REF=1` | Force GatedDeltaNet's decomposed Loop path even for the scalar-gate case that would otherwise use the fused op. Lets you A/B fused vs decomposed on a full model, and is used by the `GatedDeltaNetRefMultiHead` unit test. |
| `GGUF_FLASH_ATTN_F32=1` | Keep flash-attn SDPA (q/k/v/mask/scale) in f32 instead of the default f16. Tests whether the f16 attention cast is a precision culprit. NOTE: the plugin's f32 *inference-precision hint* (`GGML_OPENVINO_FORCE_F32`) cannot undo this — the cast is an explicit `Convert` op baked into the graph, so it must be gated in the converter, not the plugin config. |

Consider adding a similar `getenv` seam whenever an op has two conversion strategies —
it turns a rebuild-per-experiment into a flag flip, and doubles as a test hook.

---

## Gotchas and time-wasters

- **`operator[]` on the dynamic-dim map auto-inserts 0.** In `ggml-decoder.cpp`,
  reading `m_node_dynamic_dims[src]` for an op that never populated it silently
  inserts `0`, falsely marking ggml dim 0 as the dynamic token axis. Any newly
  handled op must set its entry (and the `default` case must set `-1`), or downstream
  reshapes bake the wrong axis. Symptom: an output shape with the prefill token count
  baked into the wrong dimension.
- **Per-subgraph vs shared runtime state.** A model that splits into many subgraphs
  per token step (e.g. qwen3-next's alternating attention/linear layers) must not
  share single-value bookkeeping across them. `stateful_kv_size` had to move from the
  shared `ov_runtime_context` to the per-subgraph `decoder_runtime_ctx`; a shared
  counter advanced once per subgraph and mis-sliced state on the second one. Symptom:
  a ROI/`roi_end <= max_dim` error at the *second* decode step, not the first.
- **Big models OOM in the frontend, not just in RAM for weights.** Quantized types
  that are in the backend's `supported_types` get materialized as OV constant nodes
  *in addition to* ggml's copy, roughly doubling peak memory during compilation.
  Reducing `-c` (context) does **not** help, because the blow-up is in per-subgraph
  compilation, not the KV cache. A Q4_K model can OOM where the same-arch Q2_K runs,
  purely because Q2_K isn't in `supported_types` and stays on CPU. Exit code 137 =
  OOM-killed.
- **A "still wrong" result after forcing an op to CPU can be a *new* bug** from the
  graph re-split, not confirmation the op was innocent. See the caveat in step 3.
- **`GGML_OPENVINO=""` (empty) still enables OV.** The code tests presence, not value —
  use `env -u GGML_OPENVINO` to actually reach the CPU path (§3c-quater).
- **Iterate on a small quant, not BF16.** BF16 Qwen3.5-4B took ~25 min just to
  load/compile (single-core, ~19 GB RSS); Q4_K_M loads in seconds and reproduces the
  same structural bugs. Use the small quant for the whole loop; only reach for higher
  precision if the same-weights test (step 1) actually implicates quantization.
- **Piped `llama-cli` output never flushes.** `llama-cli ... | tail`/`| grep` buffers
  and you get an empty file. Redirect to a file (`> log 2>&1`) and grep the file. `cd`
  does not persist between shell calls — use absolute paths. `pkill -9 -f llama-cli`
  between runs so stale processes don't starve CPU cores.
- **An old reference build can predate the arch.** `build-ref` (Jun 19) *segfaults* on
  qwen3.5 because it lacks the arch entirely — it cannot serve as the reference. When the
  reference crashes on load, use the *same* `build-ov` binary with `env -u GGML_OPENVINO`
  as the CPU reference instead (§3c-quater).

---

## Checklist

1. Same-weights greedy: reference CPU vs OV. Wrong on OV only → graph bug, not quant.
2. If the *text* is bad but the graph runs: **same binary, `env -u GGML_OPENVINO` vs
   `GGML_OPENVINO_DEVICE=CPU`** (§3c-quater). Identical bad text = shared bug (llama.cpp
   build/tokenizer/template), stop debugging OV. Different = real OV bug.
3. `-n 1` vs `-n 12`: prefill bug or decode/state bug?
4. **First-divergence diff** (`llama-eval-callback`, positional sum-diff) → the first
   node that differs beyond rounding. The single highest-yield localization step.
5. Full-depth per-layer sweep — but with **element-wise cosine**, not Min/Max/Mean
   (§3c-ter). Cosine cliff = localized structural bug; smooth slope = real drift.
   **Never conclude "drift" from a permutation-invariant statistic.**
6. At the cliff layer: **full-element dumps** (`GGML_OPENVINO_DUMP_TENSOR` vs
   `GGML_DUMP_TENSOR`), then **multiset** compare (§3e). Sorted-match high + position
   cosine low = layout/wrong-source bug, not math. Match the OV tensor's distribution
   against *several* candidate reference tensors to name the source it actually read.
   Only persistent buffers (caches, subgraph boundaries) dump reliably; one at a time.
7. `GGML_OPENVINO_DISABLE_TYPES=<all quant>` → if still wrong, bug is structural (f32
   path), not any quant kernel. Then narrow types/ops from there.
8. `GGML_OPENVINO_DISABLE_OPS` to force a suspect op to CPU; binary-search. Restored
   accuracy = culprit; new crash = inconclusive (re-split artifact).
9. If a `VIEW`/reshape is implicated: inspect its `nb[]` strides + `op_case`, not just
   `ne[]` (§3e, archetype D). A strided view classified as contiguous scrambles heads.
10. Reproduce at op level in `test_ops.cpp` with **real dimensions** (H, GQA, T, B not
    collapsed), grounded in a **ggml-CPU oracle**, never a hand-derived reference.
11. Assert every conversion path of the op against that oracle; leave the test as a
    regression guard.
12. **Cross-architecture regression check.** After any shared-path fix (especially a
    VIEW/`op_case` predicate), re-run every other supported model and confirm its
    classification/output is unchanged. A guard that fixes arch A can silently reroute
    arch B's legitimate views.

> Verify, don't assume: the first visible divergence is often benign (kernel
> rounding). Only a divergence you can *eliminate* and thereby fix the output is the
> real culprit.
> And: a permutation-invariant metric (mean/sum/min/max) cannot see a layout bug — if
> it reports "diffuse drift," re-check with element-wise cosine before believing it.
```
