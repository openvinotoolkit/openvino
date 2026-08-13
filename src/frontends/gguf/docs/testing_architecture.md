# GGUF Frontend — Testing Architecture

Design proposal, partially implemented. **T0 and T1 are now built** — see §11 for exactly what
landed and what its measured cost is. Everything above T1 is still a proposal. Markers below:
**[exists]** was already there, **[done]** landed with the T0/T1 work, and anything still marked
**gap** is not implemented.

## 1. What is actually being tested

Three repositories are involved, but only **one** of them holds shared logic. Everything else is a
consumer of it:

```
                    ┌──────────────── OpenVINO ─────────────────┐
                    │  op translators + normalization passes     │  ← the only shared code
                    │  GgufDecoder  (contract, PUBLISHED header) │
                    └────┬─────────────────────────────┬─────────┘
        implements ──────┘                             └────── implements
   GgufBuilderDecoder (in OpenVINO)            GgmlOvDecoder (in llama.cpp)
   src/frontends/gguf/src/builder/             ggml/src/ggml-openvino/
            │                                             │
   FrontEnd::convert(".gguf")                   ggml-openvino backend
            │                                             │
   openvino.genai                               llama-completion / llama-bench
   MakeStateful + AdaptToGenAI                  LlamaCppToStateful
   + tokenizer from rt_info
```

That shape gives **four seams**, and a test that does not name its seam is not a test of anything in
particular:

| | Seam | Fails as |
|---|---|---|
| **S1** | ggml op semantics ↔ OV translator | wrong numbers for one op, every arch that uses it |
| **S2** | `.gguf` file ↔ builder graph | one arch converts wrong / not at all |
| **S3** | `GgufDecoder` contract ↔ its two implementations | the two decoders disagree; one path silently differs |
| **S4** | converted model ↔ a runtime's IO contract | model is correct but nothing can drive it |

Crossed with two axes that multiply: **architecture** (28 accepted by the builder, 101 known to
llama.cpp) × **execution mode** (stateless / stateful / genai-adapted / static).

**S3 is the only seam that spans repositories, and it is the only one with no gate at all today.**
The `beam_idx` bug lived exactly there: the builder declared an input the cgraph decoder did not, so
the two decoders produced different stateless IO. It was caught by design review, not by a test, and
a gate costing milliseconds would have caught it.

## 2. The governing constraint: no dependency cycle

OpenVINO must not depend on llama.cpp at build or test time — that is the documented reason the
native builder exists at all (see [frontend_design.md](frontend_design.md), "Why the native path
does not use llama.cpp"). The testing architecture must not smuggle that dependency back in.

The rule that follows: **no llama.cpp dependency in the product, or in anything that has to build or
run for the frontend to be usable.**

- ggml op oracles → pregenerated `.npy` committed under `tests/test_data/`. **[exists]**
  ([gen_ggml_reference.c](../tests/gen_ggml_reference.c) is run by hand and its output committed.)
- per-arch model fixtures → a committed *manifest* of reviewed expectations; the fixture files
  themselves are generated in the Linux CI job from a pinned llama.cpp (§4a).
- the real `GgmlOvDecoder` pairing → tested **in llama.cpp**, against a contract suite OpenVINO
  *publishes* (§6).

Two places deliberately do build llama.cpp, both test-only and both fenced:

- the **arch-fixture generation step** in the Linux build job, at a pinned commit (§4a). It produces
  test data, is skipped when the GGUF component is unaffected, and its absence degrades to a skipped
  test suite rather than a broken build — which is what keeps it out of the product's dependency graph.
- a **nightly canary job** that builds llama.cpp against OpenVINO master (§8).

## 3. Tiers, and the cost principle

Each defect class should be caught by the cheapest tier capable of catching it. Today most classes
are caught by the most expensive one — a human running a manual sweep over a local model zoo — which
is why the seven defects recorded in the 2026-07-28 arch sweep were found all at once, late, by hand.

| Defect class | Cheapest catcher | Cost | Today |
|---|---|---|---|
| wrong op formula | T0 op unit | ms | **[done]** all 58 registered ops, gated |
| wrong dequant for a quant type | T0 dequant unit | ms | **[exists]** `test_dequant_vs_ggml.cpp` |
| conversion/pass contract broken | T1 graph unit | ms | **[exists]** `test_extensions.cpp` |
| arch X stops converting | T1 synthetic fixture | 2.5 ms | **[done]** `test_arch_conversion.cpp`, 101 archs |
| graph for arch X changed unintentionally | T1 fingerprint | 2.5 ms | **[done]** pinned per converting arch |
| the two decoders disagree | T2 contract | ms | **gap** — nothing |
| wrong numerics for arch X | T3 logits vs ggml CPU | seconds | **gap** — manual |
| tokenizer / E2E text | T4 | minutes | **[exists]** partial, opt-in |
| perf / memory regression | T5 | minutes | **gap** — ad-hoc shell scripts |

### T0 — op and kernel units (hermetic, OpenVINO)
Single-op models through `SingleOpDecoder`, checked against committed ggml outputs. Already the
strongest tier, and now complete with respect to the op table:
- **[done] Every op registered in `op_table.cpp` is converted by some test**, enforced by
  `test_op_coverage.cpp`. The "tested" side of that comparison is collected at run time from
  `SingleOpDecoder`'s constructor, so it cannot drift from what the tests actually do. Closing the
  last 9 ops found a real defect: `GGML_UNARY_OP_GELU_QUICK` was implemented as tanh-GELU instead of
  ggml's `x*sigmoid(1.702x)` — off by 2.2e-2, and by ~7 orders of magnitude in the negative tail.
  Nothing had converted the op, and the correct reference data was already committed but unread.
- **[done]** The real-ggml `.npy` pairs are wired in as a second, independent check
  (`GGUFUnaryVsGgml`): the parameterized cases encode the ggml formula by hand in C++, these run the
  actual kernel's captured output, so a misreading cannot be baked into both sides. The tolerances
  are set by ggml's own fp16 GELU lookup table (~2e-3 / ~3.3e-3), not by OV.
- **Gap:** reference generation is still a manual two-step (`gen_ggml_reference.c` → `.py` →
  `.npy`). Make it one scripted target with a recorded llama.cpp commit hash, the way
  `gen_arch_fixtures.py` now does for T1(b).
- The oracle must stay *real ggml*, never a numpy reimplementation of the formula — a numpy oracle
  encodes the same misreading the translator might have (this is exactly the trap `GELU_QUICK` fell
  into). Op tests therefore either compute their reference inline, where the semantics are
  unambiguous, or compare against captured ggml output; there is deliberately no numpy formula
  oracle in the tree.

### T1 — graph and contract units (hermetic, OpenVINO)
Two kinds, both millisecond-cheap and both precommit:

**(a) Pass/mode contracts** — what `test_extensions.cpp` does now: stateless is the default,
`MakeStateful` as a `DecoderTransformationExtension` swaps the mode, `skip_caches` works, the
stateless graph's inputs are exactly the decoder's inputs, `MakeStateful` adds exactly `beam_idx`.
Keep and extend; this is the tier that should own every IO-contract invariant.

**(b) Per-arch conversion over synthetic fixtures** — **[done]**, and it was the largest single
coverage win available. `test_arch_conversion.cpp`, §4a.

### T2 — cross-decoder equivalence (S3)
The invariant: *for one model, both decoders produce the same stateless graph.* Decomposes into two
claims that need different homes:

- **Neither decoder invents or omits an input.** Testable in OpenVINO with a test-double second
  decoder — `SplitIoDecoder` in `test_extensions.cpp` covers the
  `get_model_inputs`/`get_model_extra_inputs` split half. **[exists]**
- **The real `GgmlOvDecoder` agrees with `GgufBuilderDecoder` on a real file.** Not expressible in
  OpenVINO without the forbidden dependency. Belongs in llama.cpp: it has both decoders in-process
  (it links `openvino::frontend::gguf` and can drive it on the same path). Compare
  graph fingerprints, not text. §6.

### T3 — numerics per architecture (real models, nightly)
Oracle is **llama.cpp's default plain ggml CPU backend** on the same `.gguf`. Compare logits, not
generated text: text comparison is a lossy proxy that only fails after drift has already compounded,
and greedy streams legitimately diverge late. NMSE on the first-position logits vector is a sharp,
threshold-able signal — this is exactly what `test-llama-archs` already computes (§6).

Generated-text checks stay useful as a *coarse* smoke gate (gibberish is unmistakable) but must not
be the primary numerical assertion.

### T4 — end-to-end product (real models, nightly)
GenAI `LLMPipeline` on a `.gguf`: tokenizer built from `rt_info`, `MakeStateful` + `AdaptToGenAI`,
sampling, chat template. **[exists]** — `test_gguf_reader.py` in precommit,
`test_cli_text_gguf.py` (WWB similarity vs llama-cpp-python) opt-in behind `WWB_GGUF_TESTS=1`.
Gap: the WWB suite is not wired into any scheduled job, so in practice it runs when someone
remembers to run it.

### T5 — performance and memory (real models, nightly)
Fixed small model set; record TTFT / TPOT / peak *anonymous* memory (not RSS — the file-backed mmap
of the weights dominates RSS and is not the interesting number) against a tracked baseline, with the
llama.cpp default ggml CPU backend as the control on the same file. Measured noise floor on this
machine is ±2 %, so a threshold tighter than ~5 % will flap.

## 4. Fixtures — the load-bearing decision

Every tier above T1(a) needs models, and that is the reason none of them are automated: real GGUFs
are gigabyte-scale, network- and license-encumbered, and cannot be committed. So the arch matrix is
checked by hand, on one developer's local zoo, and regressions surface in batches.

**Two fixture classes, split by what they can actually prove.**

### 4a. Synthetic tiny GGUF, in-repo, precommit — **[done]**

llama.cpp can already emit a minimal valid `.gguf` for every architecture it knows, via
`llama_model_saver`, exposed as `test-llama-archs --out <dir>`. Measured:

- **101 architectures**, 5–7 MB each, all written in **1.8 s**.
- Converted through the frontend in **~2.5 ms** each.
- **23** archs convert cleanly; each one's `{op count, input count}` is now pinned as a fingerprint.
- **`jais2` fails outright** — `MatMul` dimension mismatch, `ffn_down` fed a 192-wide operand against
  a `[256,384]` weight. A real defect in a currently-`experimental` arch, surfaced in the first
  minutes of running this. (Adds to the seven from the recorded arch sweep.) Recorded in the manifest
  as `broken`, which asserts it *still fails* — fixing it makes the test demand promotion, so the
  known-broken list cannot rot into a permanent excuse list.
- The other **77** archs are cleanly rejected as outside the accept-list — which is not a failure but
  a free, exact, machine-checkable statement of *what is not supported yet* (§5). The test asserts
  the rejection comes from the accept list specifically, so a crash or a silent wrong-graph success
  is distinguishable from "not supported".

Two things had to be handled:

- **The frontend hard-required `general.file_type`, which llama.cpp's own writer does not emit** —
  every fixture failed to load on it. That the frontend required a key llama.cpp omits was itself the
  finding; the value is *written and never read anywhere* in the frontend (every weight carries its
  own type in the tensor info, which is what the dequant path uses), so it is now optional in
  `quant/gguf.cpp` rather than injected by the generator.
- **101 × 5 MB is not shippable, and not necessary.** The whole structural input to architecture
  detection and graph construction is the GGUF *header*: the bytes before `min(tensor.data_offset)`,
  i.e. magic + KV metadata + tensor table. That is what a fixture is
  (`test_data/arch_fixtures/*.gguf.hdr`, ~25 KB each); the C++ test appends the manifest's recorded
  number of zero bytes to rebuild a loadable file. Zero-filling is exact here rather than approximate
  — every tensor the model saver writes is F32, so no dequantization is involved and no block-scale
  field can be invalidated by a zero. Verified: zero-filled, seeded-RNG and sparse reconstructions all
  convert to the identical graph as the original.

  The headers are **seed-independent** — verified byte-identical for `-s 1` and `-s 999`, since the
  seed only feeds weight values. So the fixture set is reproducible regardless of the generator's
  `std::random_device` default seed, though `gen_arch_fixtures.py` pins one anyway.

- **The headers are generated in CI, not committed.** They started out committed (2547 KB raw → 120 KB
  packed) and that works, but it puts 101 opaque binaries in the tree that no reviewer can read and
  that have to be regenerated by hand whenever llama.cpp adds an architecture. Instead, the Linux
  build job clones llama.cpp at a pinned commit, builds only `test-llama-archs`, and runs
  `gen_arch_fixtures.py --fetch` into the tests artifact — **37.6 s** end to end, verified to
  reproduce all 101 committed headers byte-identically. What stays committed is `manifest.txt`: the
  reviewed expectation per architecture, which is the part that is source rather than output.

  **The pin is load-bearing, not hygiene.** `test-llama-archs` writes whatever KVs llama.cpp currently
  defines, so an upstream KV addition shifts every fixture's bytes at once. Measured: between the pin
  and a head 13 days later, all 101 headers changed (three new `attention.indexer.*` KVs) and two new
  architectures appeared — while the op counts stayed identical. Following upstream would therefore
  turn an unrelated llama.cpp commit into a red OpenVINO precommit; bumping the pin is its own
  reviewed change.

  Consequences, stated rather than discovered later: the arch suite runs on **Linux only** (the step
  is in `job_build_linux.yml`; mac/Windows/arm64 get the manifest without fixtures and
  `GTEST_SKIP()`), and the generation step costs one llama.cpp build per affected Linux build job.
  `test_arch_conversion.cpp` counts present fixtures rather than testing a boolean, so *no* fixtures
  is a skip and a *partial* set is a failure — an incomplete generation cannot quietly shrink
  coverage to whatever happened to be written.

What synthetic fixtures **can** prove: the arch converts; the graph is structurally what it was
(fingerprint); the IO contract holds; the accept-list is honest; both decoders agree structurally.

What they **cannot** prove, verified rather than assumed:
- **No vocabulary** — `tokenizer.ggml.model` is `no_vocab`, so the GenAI path rejects them at the
  tokenizer and T4 is out of reach. Confirmed by running one through `bench_gguf_perf`.
- Weights are random small normals in F32/F16, so **quantization-kernel accuracy is untouched** and
  any coherent-text assertion is meaningless.
- I could not drive one to a successful inference by hand-feeding the stateless inputs (a
  `ScatterUpdate` index bound, then a mask-width broadcast). That is a finding in its own right —
  **the stateless IO contract is undocumented and unexercised at runtime** — but it means T3-style
  numerics on synthetic fixtures needs a documented, tested input-feeding helper before it will work.

### 4b. Real small models, cached, nightly

Unavoidable for T3/T4/T5. Rules:
- One *smallest available* real model per **verified** arch, pinned by repo + filename + revision.
- Fetched into a CI-persistent HF cache, never per-run. The existing model-hub nightly jobs already
  mount `/mount/caches/huggingface`; reuse that mechanism rather than inventing one.
- Many of the models in the local zoo are symlinks into evicted HF blobs — a dangling-fixture check
  must run first and report *skipped* distinctly from *failed*, or the suite reports 7 phantom
  failures (observed).

## 5. One architecture registry

There are currently **three** independent lists of what works, and they can drift apart silently:
`verified_archs()` / `experimental_archs()` in `gguf_builder.cpp`, the tables in
[supported_models.md](supported_models.md), and the model list in genai's `test_cli_text_gguf.py`.

Replace with one machine-readable registry in OpenVINO, consumed by everything else:

```yaml
llama:    {builder: verified,     converts: yes, numerics: yes, fixture: <hf repo/file>}
gemma4:   {builder: verified,     converts: yes, numerics: yes, kv_precision: f16}
jais2:    {builder: experimental, converts: NO,  ticket: XXXXX}   # found 2026-07-30
exaone4:  {builder: experimental, converts: yes, numerics: NO, ticket: XXXXX}
qwen3next:{builder: unsupported}
```

Three properties make this worth doing:

1. **Known-broken does not block, but newly-broken does.** `converts: NO` is an xfail.
2. **XPASS is a failure.** An entry that starts passing fails the test, forcing the ledger to be
   updated. Without this, a ledger rots into a permanent list of excuses — the mechanism that lets
   `test-llama-archs`'s eleven `// FIXME` skips persist.
3. **`supported_models.md` is generated from it**, so the documentation cannot drift from the code,
   and the promotion path experimental → verified is a single reviewable diff.

## 6. llama.cpp's side, and the one big unexploited gate

llama.cpp already has precisely the harness T3 wants, and it is currently switched off.

`tests/test-llama-archs.cpp` enumerates every architecture, builds a tiny model, runs it on **every
registered ggml backend**, and reports **NMSE against the CPU backend** with a `1e-4` threshold plus
a GGUF round-trip check. `ggml-openvino` registers as a device and is already enumerated — it shows
up as `OpenVINO Runtime` in the device column. So the matrix "all archs × OV-vs-CPU logits" is
already written; it just does not run:

- `build-openvino.yml` runs `ctest -L main -E "test-llama-archs"` on both CPU and GPU, with
  `# TODO: fix and re-enable the test-llama-archs test below`.
- Verified why: with `GGML_OPENVINO_DEVICE=CPU` it starts the OpenVINO device row and then core-dumps.
  Cause not yet isolated.

**Re-enabling `test-llama-archs` for the OpenVINO device is the highest-value change available on the
llama.cpp side** — it converts the entire per-arch numerical sweep from a manual activity into a
precommit gate, using upstream's own harness and threshold, with no new infrastructure. It needs the
crash root-caused first, and it will need an expectations mechanism (§5) since not every arch is
expected to pass.

llama.cpp should also host:
- **The real cross-decoder equivalence test (T2/S3).** It is the only process with both decoders. Same
  file, converted both ways, graph fingerprints compared.
- **A `GgufDecoder` contract suite instantiated against `GgmlOvDecoder`.** OpenVINO publishes the
  suite as headers and llama.cpp instantiates it — the pattern already used by
  `src/frontends/tests/frontend/shared/include/*.hpp` for the other frontends. This is what makes the
  contract testable from the implementer's side without OpenVINO depending on llama.cpp.

## 7. openvino.genai's side

Owns T4 and the contract at S4 — and only that. Its GGUF surface is the *consumer* wiring:
`MakeStateful` registration, `AdaptToGenAI`, tokenizer-from-`rt_info`, KV precision.

- **[exists]** `test_gguf_reader.py` in precommit, guarded on the `GGUF` smart-CI component.
- **[exists]** `test_cli_text_gguf.py` — WWB similarity against llama-cpp-python on the same file, one
  model per arch family. Wire it into a scheduled nightly instead of leaving it behind an env var.
- **Gap:** no test asserts the *IO contract* `AdaptToGenAI` depends on. It asserts `beam_idx` exists
  and throws a clear message otherwise, which is good, but nothing pins the rest of the contract
  (`input_ids`/`attention_mask`/`position_ids` names, dtypes, ranks, the `[b,seq,vocab]` logits
  reshape). Those are graph-level facts, cheap to assert, and belong in genai's C++ unit tests —
  ideally as the *same* assertions the frontend's `adapt_to_genai.hpp` doc block states in prose.
- **Gap:** GenAI must not silently accept a stateless model. The assert exists; it needs a test.

Anti-pattern to avoid: genai's WWB similarity score becoming the de-facto detector for op-level
bugs. It is a slow, noisy, threshold-y signal at the far end of the pipeline. If a translator bug is
first noticed as a similarity drop in a nightly WWB run, the T0/T1 gates have failed.

## 8. CI placement

**OpenVINO**
- *Precommit* — T0 + T1 over synthetic fixtures. Target **< 60 s**; **[done]** the whole suite is
  **231 tests in 1.35 s**, so the budget is not a constraint at this size.
- **[done]** *Plumbing.* There was **no `GGUF_FE` component** (genai has `GGUF`; OpenVINO had
  nothing), so nothing could be scoped to frontend changes, and the `GGUF frontend tests` step in
  `job_cxx_unit_tests.yml` ran unconditionally — the only frontend step with no `if:` guard. Now:
  `GGUF_FE` in `.github/components.yml` (`revalidate: [CPU]`, `build: [CPU]` — the op tests infer on
  the CPU plugin), `'category: GGUF FE'` in `.github/labeler.yml` (which is what makes the component
  name resolve, via CI's `component_pattern: "category: (.*)"`), the `if:` guard on the test step,
  `ov_gguf_frontend_tests` in `.github/coverage/tests_cpp.yml`, and a step in
  `linux_sanitizers.yml`. Verified in both directions with smart_ci itself: a GGUF-only change
  affects 12 components including CPU build+test; a PDPD-only change yields `GGUF_FE: None`.
- **[done]** *Fixture generation (§4a)* sits in `job_build_linux.yml`, not in the test job, and that
  placement is forced: `job_cxx_unit_tests.yml` has **no source checkout and no cmake** (the `ov_test`
  image ships `clang` only). The build job has both, so it generates the fixtures into
  `$INSTALL_TEST_DIR/tests/test_data/arch_fixtures/` and they travel to the test job inside the
  existing tests artifact, like any other test data. Consequence: the arch suite is **Linux-only**, and
  the four other platform workflows calling `job_cxx_unit_tests.yml` (`ubuntu_24` shares the Linux
  build, then `linux_arm64`, `mac_arm64`, `windows_vs2022_release`) skip it.
- *Nightly* — T3 numerics on cached real small models; T5 perf/memory; the **llama.cpp canary**: clone
  a pinned llama.cpp, build `ggml-openvino` against freshly-built OpenVINO, run its OV tests. Today
  llama.cpp CI pins OpenVINO **2026.2.1 release archives**, so a breaking change to
  `decoder.hpp` — a *published* header two repos compile against — is invisible until someone bumps
  the pin. The signal belongs where the breaking change lands.

**llama.cpp** — `build-openvino.yml` exists and runs `ctest -L main` on CPU and GPU. Add:
re-enabled `test-llama-archs` for the OV device (§6), the cross-decoder equivalence test, the
instantiated contract suite.

**openvino.genai** — existing precommit; promote the WWB gguf suite to a scheduled nightly; add the
S4 IO-contract units.

## 9. Techniques worth productizing

Two ad-hoc methods used during recent debugging are more valuable as standing gates:

- **Graph-neutrality proof.** A change meant to be graph-neutral (refactor, relocation, renaming)
  must leave a byte-identical `.bin` and a structurally identical `.xml`. This is how the `beam_idx`
  relocation was verified, by hand. [`graph_fingerprint.py`](../tests/graph_fingerprint.py) already
  computes the right thing but is gated behind `GGUF_FINGERPRINT_MODELS` because it needs real
  models. **With synthetic fixtures it can run unconditionally in precommit**, and every refactor
  proves its own neutrality for free. Note that raw `.xml` diffing is useless without normalizing
  auto-generated node counters — removing one early `Parameter` renumbers everything after it and
  produced an 11,352-line semantically-empty diff.
- **Differential seams.** The `DISABLE_OPS` / `DISABLE_TYPES` / `DEBUG_OUTPUT` / eval-callback
  first-divergence machinery documented in
  [debugging_accuracy.md](debugging_accuracy.md) is a debugging aid today. The
  first-divergence comparison in particular is a *test* shape: when a T3 NMSE check fails, the suite
  should automatically report the first diverging node rather than only the final logits delta.

## 10. Sequencing

Ordered by value per unit of work, not by tier number.

1. **Arch registry (§5)** + generate `supported_models.md` from it. Unblocks everything else and
   immediately records `jais2` and the seven known-broken archs as data instead of prose. The
   `manifest.txt` from item 2 is now a partial, machine-checked stand-in for the `converts:` column,
   so the registry's remaining job is the `numerics:`/`fixture:` columns and doc generation.
2. **[done] Synthetic fixture generator + manifests (§4a)**, and the per-arch conversion test over
   them. Generated in the Linux build job from a pinned llama.cpp (37.6 s), 255 ms to run 101 archs in
   precommit, nothing binary in the tree. `general.file_type` made optional.
3. **[done] `GGUF_FE` smart-CI component (§8)**, plus labeler entry, test-step guard, coverage
   config, and sanitizers step.
4. **Root-cause and re-enable `test-llama-archs` for the OV device (§6).** Highest single-item value;
   turns the whole per-arch numerical sweep into a gate. **Now the top remaining item.**
5. **[done] Close the untested ops (§T0)** and add the completeness check — found the
   `GELU_QUICK` defect.
6. **Cross-decoder equivalence in llama.cpp (§6)** — closes the seam the `beam_idx` bug came through.
7. **Real-model nightly (T3/T4/T5)** with the dangling-fixture pre-check and tracked baselines.
8. **Published `GgufDecoder` contract suite (§6)**; llama.cpp instantiates it.
9. **llama.cpp canary in OpenVINO nightly (§8)** — closes the version-skew hole.

## 11. What landed, and what it cost

| | Where | Tests | Time |
|---|---|---|---|
| op completeness gate | `tests/test_op_coverage.cpp` | 2 + 1 global gate over 58 ops | ~0 ms |
| new unary op units | `tests/test_ops.cpp` (`GGUFUnary`, +6 cases) | 10 | 4 ms |
| real-ggml unary oracle | `tests/test_ops.cpp` (`GGUFUnaryVsGgml`) | 3 | 2 ms |
| permute / view / topk units | `tests/test_ops.cpp` | 7 | 9 ms |
| per-arch conversion | `tests/test_arch_conversion.cpp` | 101 + 1 manifest guard | 255 ms |
| **whole binary** | `ov_gguf_frontend_tests` | **231** | **1353 ms** |

Plus one CI step: fixture generation in the Linux build job, **37.6 s**, gated on `GGUF_FE.test`.
Without it the 101 arch tests skip and the other 130 are unaffected.

Two product fixes came out of it: `GGML_UNARY_OP_GELU_QUICK` had the wrong formula
(`src/op/unary_math.cpp`), and `general.file_type` was a hard requirement for a value the frontend
never reads (`src/quant/gguf.cpp`).

### The arch fixtures

CI generates them: the "Generate GGUF arch fixtures" step in
[job_build_linux.yml](../../../../.github/workflows/job_build_linux.yml) runs

```
python3 src/frontends/gguf/tests/gen_arch_fixtures.py --fetch --out-dir <tests-install>/test_data/arch_fixtures
```

which shallow-fetches the pinned llama.cpp commit, builds only `test-llama-archs`, emits all archs,
truncates each to its header, and rewrites `manifest.txt`. To make the suite live in-tree locally,
run the same thing with `--llama-src <checkout>` (or `--fetch`) and no `--out-dir`; the output is
gitignored. Four deliberate behaviours:

- It **preserves existing expectations** and defaults every *new* fixture to `reject`. The manifest
  therefore never asserts "whatever the frontend currently happens to do" — promoting a new arch to
  `convert` is a one-word, reviewable edit, and its fingerprint must be added by hand. This is why the
  generated fixtures can be trusted without review: the *expectations* were reviewed once, in the
  manifest, and regeneration cannot change them.
- `LLAMA_CPP_COMMIT` pins upstream and must be a commit on `ggml-org/llama.cpp` master, since
  `--fetch` fetches from there. Bumping it is a reviewed change of its own — see §4a for why.
- It builds with `LLAMA_BUILD_TESTS=ON` and everything else off. That flag is mandatory:
  `test-llama-archs` is a test target, and upstream's release builds set it `OFF`, which is why no
  prebuilt llama.cpp package ships the binary and `apt install` is not an option.
- The step is gated on `GGUF_FE.test`, so PRs that do not touch the frontend pay nothing for it.

**Seven of the 28 builder-supported archs have no fixture**, because llama.cpp will not emit them:
`llama_model_saver_supports_arch` (a deny-list in `src/llama-model-saver.cpp`) or
`test-llama-archs`'s own `arch_supported` skips deepseek2-ocr, exaone-moe, gemma3, gemma4,
llama-embed, mellum and plamo3. Investigated: the saver has **no per-arch code at all** — it writes
one flat KV list for every arch — so "unsupported" never means "cannot be serialized". The deny-list
means "does not survive a bit-exact save→load→infer round-trip", which is a stricter property than
this suite needs. Un-gating them locally, **6 of the 7 save and convert fine**; only deepseek2-ocr
genuinely fails, on a missing `expert_feed_forward_length` KV in the fixture builder. So the real
coverage ceiling is 23 of 24 convertible archs, and reaching it needs an upstream fix rather than a
local patch. `gemma4` being among the seven is the notable gap: it is a *verified* arch.
