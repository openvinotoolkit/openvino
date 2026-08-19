# GGUF Frontend — Testing Architecture

How tests for this frontend are organized, and where a new test belongs. T0 and T1 (below) are
implemented; everything above T1 needs real `.gguf` models and is largely manual today — see
§5 "What's missing" for the gaps and how they're expected to close.

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

That shape gives four seams, and a test that does not name its seam is not a test of anything in
particular:

| | Seam | Fails as |
|---|---|---|
| **S1** | ggml op semantics ↔ OV translator | wrong numbers for one op, every arch that uses it |
| **S2** | `.gguf` file ↔ builder graph | one arch converts wrong / not at all |
| **S3** | `GgufDecoder` contract ↔ its two implementations | the two decoders disagree; one path silently differs |
| **S4** | converted model ↔ a runtime's IO contract | model is correct but nothing can drive it |

Crossed with two axes that multiply: **architecture** (28 accepted by the builder, 101 known to
llama.cpp) × **execution mode** (stateless / stateful / genai-adapted / static).

**S3 is the only seam that spans repositories, and today it has no automated gate.** The `beam_idx`
bug lived exactly there: the builder declared an input the cgraph decoder did not, so the two
decoders produced different stateless IO. It was caught by design review, not by a test.

## 2. The governing constraint: no dependency cycle

OpenVINO must not depend on llama.cpp at build or test time — that is the documented reason the
native builder exists at all (see [frontend_design.md](frontend_design.md), "Why the native path
does not use llama.cpp"). The testing architecture must not smuggle that dependency back in:

- ggml op oracles → pregenerated `.npy` committed under `tests/test_data/`
  ([gen_ggml_reference.c](../tests/gen_ggml_reference.c) is run by hand and its output committed).
- per-arch model fixtures → a committed *manifest* of reviewed expectations; the fixture files
  themselves are generated in the Linux CI job from a pinned llama.cpp (§4).
- the real `GgmlOvDecoder` pairing (S3) → belongs in llama.cpp, against a contract suite OpenVINO
  would publish (not yet done).

Two places deliberately do build llama.cpp, both test-only and fenced off from the product: the
arch-fixture generation step in the Linux build job (pinned commit, skips cleanly when the GGUF
component is unaffected — see §4), and a nightly canary that would build llama.cpp against OpenVINO
master (not yet done).

## 3. Tiers, and the cost principle

Each defect class should be caught by the cheapest tier capable of catching it.

| Tier | What | Cost | Status |
|---|---|---|---|
| T0 | op/kernel units, real-ggml `.npy` oracle, hermetic | ms | done — all registered ops, gated by `test_op_coverage.cpp` |
| T1(a) | pass/mode contract units (`test_extensions.cpp`) | ms | done |
| T1(b) | per-arch conversion over synthetic fixtures | ms | done — 101 archs, ~2.5 ms each |
| T2 | cross-decoder equivalence (S3) | ms | gap — needs a home in llama.cpp, see §5 |
| T3 | numerics per arch vs ggml CPU oracle, real models | seconds | gap — manual today |
| T4 | end-to-end product (GenAI `LLMPipeline`, tokenizer, sampling) | minutes | partial, opt-in |
| T5 | perf/memory regression, real models | minutes | gap — ad-hoc local scripts |

**T0 — op and kernel units (hermetic, OpenVINO).** Single-op models through `SingleOpDecoder`,
checked against committed ggml outputs. `test_op_coverage.cpp` asserts every op registered in
`op_table.cpp` is exercised by some test — closing the last gaps this way caught a real bug
(`GGML_UNARY_OP_GELU_QUICK` was implemented as tanh-GELU instead of ggml's `x*sigmoid(1.702x)`,
off by 2.2e-2). The oracle must stay *real ggml*, never a numpy reimplementation of the formula —
a numpy oracle can encode the same misreading the translator made.

**T1 — graph and contract units (hermetic, OpenVINO).** (a) Pass/mode contracts: stateless is the
default, `MakeStateful` as a `DecoderTransformationExtension` swaps the mode, the stateless graph's
inputs are exactly the decoder's inputs. (b) Per-arch conversion over synthetic fixtures — the
largest single coverage win available; see §4 below for how the fixtures are produced.

**T2 — cross-decoder equivalence (S3).** The invariant: for one model, both decoders produce the
same stateless graph. Splits into "neither decoder invents or omits an input" (testable in
OpenVINO today via a test-double decoder, `SplitIoDecoder` in `test_extensions.cpp`) and "the real
`GgmlOvDecoder` agrees with `GgufBuilderDecoder` on a real file" (not expressible in OpenVINO
without the forbidden llama.cpp dependency — belongs in llama.cpp, comparing graph fingerprints).

**T3 — numerics per architecture, real models.** Oracle is llama.cpp's default plain ggml CPU
backend on the same `.gguf`. Compare logits (NMSE on first-position logits), not generated text —
text is a lossy proxy that only fails after drift has already compounded. Generated-text checks
stay useful as a coarse smoke gate only. Until this tier is automated,
[`compare_with_llama.py`](../tests/compare_with_llama.py) is the ad-hoc local stand-in: it
greedy-decodes a prompt through the frontend's own stateless gguf-IO contract (round-tripping the
KV cache itself, via `gguf_io_utils.py`) so the output can be diffed against `llama-simple`.

**T4 — end-to-end product, real models.** GenAI `LLMPipeline` on a `.gguf`: tokenizer built from
`rt_info`, `MakeStateful` + `AdaptToGenAI`, sampling, chat template. `test_gguf_reader.py` runs in
precommit; `test_cli_text_gguf.py` (WWB similarity vs llama-cpp-python) is opt-in behind
`WWB_GGUF_TESTS=1` and not yet wired into a scheduled job.

**T5 — performance and memory, real models.** Fixed small model set; TTFT/TPOT/peak anonymous
memory (not RSS — the file-backed mmap of the weights dominates RSS and isn't the interesting
number) against a tracked baseline, with llama.cpp's default ggml CPU backend as the control.
Until this tier is automated, [`bench_gguf.py`](../tests/bench_gguf.py) is the ad-hoc local
stand-in: load/compile time and prefill/decode throughput for the frontend versus `llama-simple`.

## 4. Fixtures — the load-bearing decision

Every tier above T1(b) needs real models, and that is the reason none of them are automated yet:
real GGUFs are gigabyte-scale, network- and license-encumbered, and cannot be committed. Two
fixture classes, split by what they can actually prove:

### Synthetic tiny GGUF, in-repo, precommit

llama.cpp can emit a minimal valid `.gguf` for every architecture it knows via `llama_model_saver`
(`test-llama-archs --out <dir>`, 101 architectures, 5-7 MB each). The whole structural input to
architecture detection and graph construction is the GGUF *header* — everything before
`min(tensor.data_offset)` — so only that (~25 KB) is kept as a fixture
(`test_data/arch_fixtures/*.gguf.hdr`); the C++ test appends the manifest's recorded number of zero
bytes to rebuild a loadable file. Headers are seed-independent (weight values don't affect them),
so the fixture set is reproducible regardless of the generator's seed.

The headers themselves are **generated in CI, not committed** — the Linux build job clones
llama.cpp at a pinned commit (`LLAMA_CPP_COMMIT`), builds only `test-llama-archs`, and runs
[`gen_arch_fixtures.py --fetch`](../tests/gen_arch_fixtures.py) into the tests artifact (~38 s).
What stays committed is `manifest.txt`: the reviewed expectation per architecture — the part that
is source rather than generated output. The pin is load-bearing, not hygiene: `test-llama-archs`
writes whatever KVs llama.cpp currently defines, so an upstream KV addition shifts every fixture's
bytes at once; bumping the pin is its own reviewed change. Consequently the arch suite runs on
**Linux only** (the generation step needs a source checkout + cmake, which the other platforms'
test jobs don't have); other platforms get the manifest without fixtures and skip.

What synthetic fixtures **can** prove: the arch converts; the graph is structurally what it was
(pinned `{op count, input count}` fingerprint); the IO contract holds; the accept-list is honest.
What they **cannot** prove: real quantization-kernel accuracy (weights are random F32/F16 normals),
or anything requiring a real tokenizer (`tokenizer.ggml.model` is `no_vocab` in these fixtures).

To regenerate fixtures locally: `gen_arch_fixtures.py --llama-src <checkout>` (or `--fetch`), no
`--out-dir` — the output is gitignored.

### Real small models, cached, nightly

Unavoidable for T3/T4/T5. One *smallest available* real model per **verified** architecture,
pinned by repo + filename + revision, fetched into the model-hub tests' persistent HF cache rather
than per-run.

## 5. What's missing, and where it should live

- **One architecture registry.** There are currently three independent lists of what works —
  `verified_archs()`/`experimental_archs()` in `arch_registry.cpp`, the tables in
  [supported_models.md](supported_models.md), and genai's `test_cli_text_gguf.py` model list — and
  they can drift apart silently. A single machine-readable registry (arch → builder support,
  converts, numerics, fixture) that generates `supported_models.md` would close that gap and make
  known-broken archs explicit `xfail` entries instead of prose.
- **llama.cpp already has the T3 harness, switched off.** `tests/test-llama-archs.cpp` builds a
  tiny model per architecture, runs it on every registered ggml backend including
  `ggml-openvino`, and reports NMSE against the CPU backend — exactly what T3 needs. It's disabled
  for the OpenVINO device in `build-openvino.yml` (`# TODO: fix and re-enable`) because it
  currently crashes; root-causing that and re-enabling it is the single highest-value remaining
  item, since it turns a manual per-arch sweep into a precommit-equivalent gate for free.
- **T2 (cross-decoder equivalence) and a published `GgufDecoder` contract suite** belong in
  llama.cpp, the only process with both decoders in one binary — compare graph fingerprints on the
  same file converted both ways.
- **genai's WWB suite (T4)** should move from an opt-in env var to a scheduled nightly, and needs
  unit-level assertions on the S4 IO contract itself (`input_ids`/`attention_mask`/`position_ids`
  names, dtypes, ranks), not just a similarity score — a translator bug should never be first
  noticed as a WWB regression.
- **Graph-neutrality as a standing gate.** A change meant to be graph-neutral (refactor,
  relocation, renaming) should leave the fingerprint unchanged.
  [`graph_fingerprint.py`](../tests/graph_fingerprint.py) computes the right thing but is gated
  behind `GGUF_FINGERPRINT_MODELS` because it needs real models today; with the synthetic fixtures
  it could run unconditionally in precommit.
- **A llama.cpp canary in OpenVINO's nightly**, since llama.cpp CI currently pins OpenVINO release
  archives rather than building against master — a breaking change to the published `decoder.hpp`
  is invisible until someone bumps that pin.

## CI placement today

Precommit runs T0 + T1 over synthetic fixtures (231 tests, ~1.35 s total, well under any budget).
The `GGUF_FE` smart-CI component (`.github/components.yml`, `.github/labeler.yml`) scopes the
frontend test step, coverage, and sanitizer runs to frontend changes. Fixture generation lives in
`job_build_linux.yml` rather than the test job itself, because the test job's image has no source
checkout or cmake; that's also why the arch suite is Linux-only.
