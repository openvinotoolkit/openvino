# KleidiAI ↔ oneDNN integration: OpenVINO findings & gap-fix report

**Author:** Aleksandr Voron (OpenVINO CPU plugin)
**Context:** Evaluating KleidiAI as the ACL replacement in the OpenVINO oneDNN fork on AArch64.
**Branch shared:** oneDNN `kleidiai-cherry-pick` (fork base `f82d833de6`).
**Reference platform:** Apple arm64 (macOS), FP16 models, 6 threads.

## TL;DR for the KleidiAI team

We ported KleidiAI into OpenVINO's oneDNN fork (replacing ARM Compute Library) and brought several
FP16 CV models to functional + performance parity. Starting from the 4 commits on
`jondea/replace-acl-with-kleidiai` (`9d2436344…` et al.), we found and fixed **five gaps** while
integrating. The three most relevant to KleidiAI itself:

1. **`kai_wino_convolution` recomputes constant-weight transforms on every execute** — a
   pack-once/execute-many violation that cost ~35 ms/inference on yolo_v4 (winograd conv −44% once
   cached). *This is the highest-impact finding and likely affects any oneDNN primitive built on the
   current KleidiAI conv wrappers.*
2. **`kai_convolution` rejected all fused post-ops** (ReLU/bounded-ReLU), even though KleidiAI's GEMM
   epilogue supports them — every activated conv fell back to reference (aclnet: 12/13 convs on
   reference).
3. **KleidiAI's f16 depthwise kernels exist but were not wired into any oneDNN primitive** — grouped
   depthwise convs fell back to reference (aclnet: 8/8 group-convs on reference).

A residual, un-closed gap remains: **the KleidiAI a64 f16 GEMM micro-kernels are ~1.8× slower than
ACL's for yolo_v4's 1×1/3×3 shapes** even after all setup overhead is removed — this is raw kernel
throughput and needs attention on the KleidiAI side (details in §"Open gap").

---

## Branch layout

The branch has two groups of commits. Group A is the KleidiAI team's work, cherry-picked and adapted
to our fork. Group B is what we added during integration to close functional/perf gaps — this is the
part most useful for polishing the KleidiAI-in-oneDNN story.

### Group A — cherry-picked from `jondea/replace-acl-with-kleidiai` (adapted)

| commit | summary | notes |
|---|---|---|
| `8f7d05bdca` | cpu: aarch64: add kai matmul, ipa and conv | Cherry-pick of `9d2436344f…`. Adds `kai_convolution`, `kai_wino_convolution`, `matmul/kai_matmul`, `matmul_inner_product`, `kai_utils`, KleidiAI submodule, CMake + isa-traits wiring. |
| `0dc86b75f5` | cpu: aarch64: remove ACL | Cherry-pick of `685713ad…`. Removes all ACL from oneDNN. |
| `1b511f6e11` | aarch64: fix temp kleidiai submodule | Submodule URL/path. |
| `ee9160b8e0` | cpu: aarch64: add blocked ip support for wtag=any | Cherry-pick of `aca6e5ae…`. |

**Adaptation notes (fork-specific, not KleidiAI issues, but useful to know):** our fork kept ACL under
`src/cpu/acl/` (namespace `dnnl::impl::cpu::acl`, macro `CPU_INSTANCE_ACL`) rather than upstream's
`src/cpu/aarch64/`; we used the `aarch64` namespace throughout. Our fork is oneDNN 3.10.2, so
`create_nested_grantor` (3.13) isn't available — `matmul_inner_product` uses the `nested_scratchpad_t`
idiom instead. OpenVINO keeps its **own** standalone ACL executors (separate from oneDNN), so we build
ComputeLibrary independently and force `DNNL_USE_ACL=OFF`.

### Group B — integration gap-fixes (the substance of this report)

| commit | gap class | model impact |
|---|---|---|
| `0a35798d28` enable kai convolution | dispatch/naming | conv not selected → selected |
| `0d13401c1b` fuse ReLU/bounded-ReLU post-op in kai convolution | functional | aclnet convs 1→13 on KleidiAI |
| `2926b3ca04` add KleidiAI depthwise convolution primitive | coverage | aclnet 8 group-convs → KleidiAI |
| `f630b343d0` cache constant winograd weights | **performance** | yolo_v4 winograd conv 84.7→48 ms |
| `5b3a571d7b` name kai matmul impl "kleidiai" | dispatch/naming | matmul/IP execType now identifiable |

---

## Detailed findings

### 1. Impl-info-string naming convention (`0a35798d28`, `5b3a571d7b`)

**Symptom:** even when a KleidiAI primitive's `pd_t::init()` succeeded, the conv/matmul was silently
dropped and the framework fell back to reference (`ref_any_f16`).

**Cause:** OpenVINO selects among oneDNN primitive descriptors by parsing the oneDNN
*impl-info-string* and matching it (exact match) against a per-node priority list. The cherry-picked
kernels were named `indirect_gemm:kai`, `im2row:kai`, `direct_1x1:kai`, `wino:arm`, and `"kai"` (matmul).
OpenVINO's parser recognizes a `kleidiai` token; `:kai`/`:arm` were parsed as bare `gemm`/`winograd`
(or `unknown`), which are absent from the node priority lists → dropped.

**Fix:** rename all KleidiAI impl-info-strings to carry the `kleidiai` token
(`…:kleidiai`, `"kleidiai"`). OpenVINO then canonicalizes them to `gemm_kleidiai`/`winograd_kleidiai`.

**Insight for KleidiAI/oneDNN:** the impl-info-string is a *public contract* that downstream
frameworks pattern-match on. It would help to (a) standardize on a single stable token (e.g.
`kleidiai`) across all KleidiAI kernels in the oneDNN integration branch, and (b) document it, so
integrators don't have to reverse-engineer it from `ref` fallbacks. The current mix of `:kai`,
`:arm`, `"kai"` is easy to get wrong and fails silently (no error — just slow).

### 2. Fused post-ops rejected by `kai_convolution` (`0d13401c1b`)

**Symptom:** on aclnet, 12 of 13 convolutions ran on reference; only the single conv with no fused
activation used KleidiAI.

**Cause:** `kai_convolution_fwd_t::pd_t::init()` gated on
`attr()->has_default_values(skip_mask)` where the skip-mask did **not** include `post_ops`. So any
conv carrying a fused eltwise (here: a plain ReLU, extremely common in CV models) was rejected before
kernel creation — a silent `unimplemented` with no verbose line.

**Fix:** allow `post_ops` in the skip-mask and translate a supported post-op chain into KleidiAI's
native GEMM-epilogue activation (`kai::ops::Activation`): empty → none, single `eltwise_relu`
(α=0) → ReLU, single `eltwise_clip`(0,β) → BoundedReLU(β). Anything else (sum, binary, leaky/scaled
eltwise, chains >1) is declined so it falls back to reference rather than silently dropping the op.
Verified byte-behaviour: ReLU'd outputs clamp, non-activated conv retains negatives; accuracy within
FP16 noise.

**Insight for KleidiAI/oneDNN:** the KleidiAI GEMM epilogue already supports ReLU/BoundedReLU — the
gap was purely that the oneDNN wrapper didn't advertise/consume post-ops. Since fused activation is
near-universal in inference conv graphs, the integration branch's conv/matmul wrappers should handle
the common eltwise post-ops out of the box (ReLU, ReLU6/bounded-ReLU, and ideally the eltwise+sum
"conv+add+relu" residual pattern). A model with fused activations otherwise sees ~0% KleidiAI
coverage. Consider also emitting a verbose reason when a post-op is the cause of a decline — silent
fallback made this expensive to diagnose.

### 3. Depthwise f16 not wired into oneDNN (`2926b3ca04`)

**Symptom:** aclnet's 8 GroupConvolutions (all depthwise 3×3) ran on reference.

**Cause:** `kai_convolution_fwd_t::init()` hard-rejects `with_groups()`, and there was no other
AArch64 f16 depthwise primitive in oneDNN. KleidiAI **does** ship f16 depthwise kernels
(`experimental/ops/kai/ops/conv/depthwise/depthwise_fp16.cpp`, `DepthwisePlanar<__fp16>` /
`DepthwiseDepthfirst<__fp16>`, built into `kleidiai_ops`), but nothing in the integration branch
instantiated them.

**Fix:** new `kai_depthwise_convolution_fwd_t` primitive backed by
`kai::ops::depthwise::depthwise<__fp16>()`. It accepts depthwise (channel-multiplier 1), f16, NHWC
activations + `hwigo` weights (which maps directly to KleidiAI's expected **HWIO** packing — a handy
coincidence that avoided a repack), and reuses the post-op→activation translation from finding #2.
Result: 8/8 group-convs on KleidiAI, output within FP16 noise of reference (mean rel-diff ~8e-4).

**Insight for KleidiAI/oneDNN:** the depthwise kernels are production-quality but "orphaned" in the
integration branch — adding an oneDNN depthwise-conv primitive would make them reachable for every
integrator, not just those who write the wrapper themselves. Depthwise 3×3 is ubiquitous
(MobileNet-family, many detectors), so this is high-value coverage. One ergonomics note: the
`DepthwiseArgs` layout expectations (HWIO, channel_multiplier semantics) took some reading to confirm
against oneDNN's grouped-weight descriptors — a short mapping doc would help.

### 4. ⭐ Winograd recomputes constant weights every execute (`f630b343d0`) — biggest perf win

**Symptom:** yolo_v4 winograd convs took 84.7 ms total vs ACL's 34.6 ms (~2.4×), uniformly across all
37 winograd nodes — the signature of fixed per-inference overhead, not slow math.

**Root cause (measured with per-stage timers, KAI_PROF):**
`kai_wino_convolution_fwd_t::execute()` performed, on **every** inference:
- the winograd-domain **weight transform** of the (constant) weights — **26.0 ms/inf**, and
- the KleidiAI **`pretranspose_B_array`** of those weights — **9.7 ms/inf**,

both writing into the *transient* oneDNN scratchpad, which is not persisted across calls. That's
~35.7 ms/inference of pure repeated work on data that never changes during inference. (For contrast,
`create_kai_gemm()` — rebuilding the kernel object each execute — measured **0.01 µs**, negligible;
our initial hypothesis that the kernel rebuild was costly was **refuted by measurement**.)

A subtlety that defeats the obvious fix: **OpenVINO re-reorders the constant weights into a rotating
scratchpad buffer, so the weights pointer differs on ~95% of executes** — a pointer-keyed "did the
weights change?" cache would miss almost every time. The values are deterministic, though.

**Fix:** compute the winograd-domain weights and the pretransposed weights **once**, on the first
execute, into **primitive-owned** buffers (mutex-guarded first-population for multi-stream safety;
freed in the destructor). Subsequent executes reuse them; only the input/gemm/output transforms
(genuinely activation-dependent) stay per-execute. Output is **byte-for-byte identical** to
recomputing every time. Result: winograd conv 84.7 → ~48 ms (−44%); total model conv 164 → 128 ms
(−22%); model median latency −21%.

**Insight for KleidiAI/oneDNN — this is the important one:** the KleidiAI convolution wrappers appear
designed around a "prepare inside execute" model, which conflicts with oneDNN's
**create-time-prepare / execute-cheap** contract and with how frameworks feed constant weights. Two
recommendations:
- Expose an explicit **weight-prepack / weight-transform step separable from execute**, with a
  clearly-documented lifetime, so integrators can run it once at primitive creation and store the
  result (oneDNN scratchpad vs. persistent buffer semantics matter here — the transient scratchpad is
  the wrong place). Ideally the API returns a packed-weights handle the caller owns.
- The same pattern almost certainly affects the **non-fixed-format `kai_convolution` path**: when
  `run_weight_reorder_` is set, its `execute()` also calls `pretranspose_B_array` every inference.
  yolo_v4 happened to use the fixed-format path (weights packed once by OpenVINO's compile-time
  reorder) so we didn't need to fix it there, but a model that doesn't hit fixed-format would pay the
  same per-execute repack. Worth auditing all KleidiAI oneDNN wrappers for "constant work inside
  execute".

---

## Open gap (NOT fixed — needs KleidiAI-side attention)

After removing all per-execute setup overhead, a real gap remains in **raw GEMM micro-kernel
throughput**:

| yolo_v4 conv (kernel->execute only, per inf) | KleidiAI | ACL |
|---|---|---|
| GEMM path (73 nodes) | 72.8 ms | 43.0 ms |
| Winograd path (37 nodes) | 40.2 ms | 34.6 ms |

The **GEMM path had no removable setup at all** (fixed-format weights, packed once) — the ~1.8×
difference is the KleidiAI a64 f16 GEMM kernels themselves being slower than ACL's for these shapes
(mostly 1×1 pointwise and 3×3, batch 1, on a 6-core Apple core). Post-fix totals: KleidiAI ~128 ms
conv vs ACL ~78 ms — the residual is almost entirely this kernel-throughput delta.

**Suggested investigation for the KleidiAI team:**
- Benchmark the selected a64 f16 GEMM micro-kernels in isolation on these shapes (M=H·W, small
  N=OC, K=IC or IC·9) vs ACL's equivalents; the 1×1-heavy YOLO backbone is a good stress case.
- Check kernel **selection heuristics** for small-batch / small-N inference GEMMs — ACL may be
  picking a better-tuned variant. It's possible KleidiAI has a faster kernel available that the
  `find_implementation`/config heuristic isn't choosing for these dimensions.
- The im2row path (5.8 ms/inf) is a secondary contributor; a fused/blocked im2col or a direct-conv
  kernel would help, but the GEMM itself dominates.

---

## Methodology notes (for reproducibility)

- All numbers: `benchmark_app -niter 20 -hint latency -report_type average_counters`, summing the
  `realTime` of Convolution rows in `benchmark_average_counters_report.csv`, KleidiAI build vs a
  prebuilt ACL reference build of the same model.
- Root-causing used oneDNN `ONEDNN_VERBOSE=dispatch/exec` plus temporary env-gated per-stage timers
  inside the KleidiAI `execute()` paths (removed before commit). Silent `unimplemented` returns
  inside the `init()` gate chain and the kernel-creation loop were the main diagnosis obstacle —
  more verbose dispatch reasons in the KleidiAI wrappers would speed this up for future integrators.
- Correctness for every fix was checked against reference and/or a recompute-every-execute build
  (blob dumps, seeded input); FP16 accumulation-order noise (~1e-3 rel) is expected because KleidiAI
  accumulates in fp16 where oneDNN reference uses f32.

## Summary of asks for the KleidiAI team

1. **Separate weight-prepare from execute** across the conv/winograd wrappers, with documented
   lifetime — the single biggest perf lever (§4).
2. **Handle common fused post-ops** (ReLU/bounded-ReLU/at least the residual add) in the conv/matmul
   wrappers' dispatch (§2).
3. **Wire the f16 depthwise kernels into oneDNN** as a first-class primitive (§3).
4. **Standardize & document the impl-info-string token** so frameworks can dispatch reliably (§1).
5. **Investigate a64 f16 GEMM kernel throughput / selection** for small-batch inference shapes (§Open
   gap) — the remaining parity gap after all integration fixes.
6. Emit **verbose decline reasons** from the wrappers — silent `unimplemented` made every gap above
   costly to find.
