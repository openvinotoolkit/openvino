# GGUF structural regression references

These are structural alarms, not runtime benchmarks or numerical equivalence tests.
Each case has exactly one reference file containing every named graph returned by
`adapt_to_genai` for its selected input modes. Text-only decoders cover token IDs;
multimodal decoders also cover embedding inputs. Everything runs offline and no checkpoint
weights are allocated.

## Coverage status

The catalog contains original-dimensional profiles for all 47 native architecture
handlers, including the separate LFM2.5 vocoder, and all 36 supported projector contracts.
It also records the unsupported Gemma3n audio identity in the combined vision checkpoint.
Tiny numerical fixtures do not substitute for original-dimensional profiles.
`GGUFGraphRegression.CatalogCoversEveryNativeHandler` and
`CatalogClassifiesEveryProjector` fail when a registered handler or projector
(plus the composite speech-generation branches) has no positive case.

Every checkpoint in the model hub lists (`tests/model_hub_tests/gguf/gguf_{models,mmproj}_*`
and `bonsai_models.json`) appears in a case's `models` as `<repository>/<filename>`.
Each checkpoint's header and provenance are recorded in `source` or `equivalent_sources`;
filenames alone are never trusted.
`test_catalog_maps_every_documented_checkpoint` enforces the mapping.
Cross-architecture determinism remains an acceptance requirement.

Cases are consolidated only when their captured named graphs, tensor descriptors and
conversion-relevant metadata match. Different headers and publisher provenance remain
in `equivalent_sources`. A consolidated checkpoint is represented by the selected recipe,
not independently recaptured on later runs; split it into a separate case if a future
builder starts using metadata that previously differed only for informational purposes.

## Artifacts

All paths below are under `tests/test_data/graph_regressions`, in ordinary Git text,
not LFS:

- `catalog.json`: case/checkpoint identities, provenance and projector contracts.
  `equivalent_sources` preserves each consolidated checkpoint's aliases and original
  header hash and provenance.
  Adapted decoder cases require `input_modes`: `["ids"]` for text-only checkpoints
  and synthetic text-only fixtures, or `["ids", "embeds"]` for vision or audio
  checkpoints. Classify each checkpoint rather than assuming an entire architecture
  has the same modalities. If its modalities are unknown, retain both modes until
  verified. Projectors and `native_only` cases omit `input_modes`.
  `native_only` records the reason when a case has no GenAI adaptation: embedding-only
  encoders (including LFM2.5 encoder, embedding and ColBERT checkpoints), stateless LFM3 D1
  models, noncausal sliding-window DFlash drafts and the standalone LFM2.5 vocoder. These cases keep a native
  stateful snapshot; adaptation errors in all other cases fail, never silently fall back.
  Remove the exemption when `adapt_to_genai` gains support for that contract.
- `schema.json`: shared versioned catalog, recipe and reference definitions.
- `recipes/<case>.json`: compact recipe version 2, one metadata entry or tensor group per
  line. Metadata maps each key to `[type, value]` (`"u32"`, `"f32[]"`, ...); arrays of at
  least 64 values may use `{"rle": [[value, count], ...]}` or, for signed ±1 arrays,
  bit-packed `{"signs": <base64>, "count": n}`. Tensors are `[name, ggml_type, shape]` or
  `[template, ggml_type, shape, layers]`, where one `{}` in the template expands over
  ascending, disjoint layer ranges such as `"0-2,4-63"`. Shapes use OpenVINO order,
  reversing GGUF's `ne[]`; source storage is derived from the GGML type. The runner
  expands this to a per-tensor descriptor before conversion.
- `references/<case>.json` (reference version 3): a `models` map with complete operation
  type/version counts, projection summaries, totals and one `graph_sha256` per named graph,
  plus the source projector identities. Histogram rows are kept one per line
  for readable diffs; other value arrays are inline. Each reference is limited to **32 KiB**.

## What is captured

The runner converts the recipe with the production frontend and builders. Weights are
descriptors that throw on payload access, so builder selection and weight layout planning
run without allocating checkpoint-sized weights. The exception is a bounded synthetic
identity `d2t` vocabulary map for draft models: its type and shape match the descriptor,
but its values are not the checkpoint's mapping. Image separator vectors (`v.image_newline`, `v.view_seperator`,
`model.image_newline`) are zero-filled the same way; only their length is hashed. After conversion, the runner calls the
production `adapt_to_genai` entry point, which makes decoders stateful and adapts their IO.
Every returned model is checked, not just the language model:

- Decoders: `ids/language` for `IDS_TO_LOGITS`. Cases with embedding inputs also check
  `embeds/language`, `embeds/text_embeddings` and any `embeds/text_embeddings_per_layer`
  for `EMBEDS_TO_LOGITS`. Each mode consumes an independent converted graph.
- Projectors: every returned key, including split vision encoders, position/merger
  graphs, audio encoders and all `audio_generation_*` components.
- Explicit `native_only` cases: `native`, after `GGUFMakeStateful`.

The model-map keys are part of comparison: adding, removing or renaming any component
fails the gate, as does a change confined to an auxiliary graph. There is no separate
raw or materialized stage. Weight loading and numerical behavior (repacking, Hadamard, ternary formats,
quantization arithmetic) are not covered here; the existing C++ conversion and accuracy
tests own them.

## Graph hash scope

`graph_sha256` is the SHA-256 of a canonical JSON graph (sorted keys, no whitespace).
The scope is meant to capture everything that can change how a plugin compiles or runs
the model, and nothing that only changes floating-point or weight values or debug metadata.
Integral and boolean Constants are kept because they usually encode shapes, axes, indices
and masks that select graph behavior.

| Included | Excluded |
| --- | --- |
| Operation type and opset version of every node, including nested bodies | Friendly names, internal tensor names and node creation order |
| Connectivity: source node and output port of each input, in input order; control dependencies | Floating-point Constant values (type and shape are kept) |
| Every output's element type and partial shape, with dynamic dimensions and bounds | Large Constants narrower than 32 bits (type and shape are kept) |
| All `visit_attributes` values: axes, modes, flags, target shapes, body port maps | Weight payloads, quantization scales and zero points |
| Integral Constant values with at most 64 elements or at least 32-bit integers, and boolean Constants (including the synthetic `d2t` map) | Shape symbols and original `d2t` values |
| Parameters, Results and Sinks in model order, with their tensor names | Tokenizer metadata and `fused_names`, `original_precision`, `WeightlessCacheAttribute` |
| ReadValue/Assign variable IDs, types and shapes | |
| GGUF weight descriptors: source tensor, GGML type, storage type/shape, group size, unpacked weight/scale/zero-point layouts and row/column plans | |
| GGUF runtime keys (`gguf*`) on nodes and the model; `gguf.draft_context` is recorded as presence only | Draft-context friendly-name strings |
| Node markers that steer plugins: `decompression`, `dequantization_node`, `keep_const_precision`, `precise`, `DisableConstantFolding`, `DisablePrecisionConversion`, `primitives_priority`, `shape_subgraph` | |

Node IDs are assigned by a depth-first walk from the Results in input-port order, so
two builders that create the same graph in a different order produce the same hash.
Capture fails on an unclassified GGUF or node-level OpenVINO runtime key and on an
unsupported attribute type, so a new marker cannot silently fall outside the scope.

The comparison has no numeric tolerance. Tolerance is expressed only through this scope:
changes to floating-point Constants (such as a new epsilon or RoPE scale), weight payloads,
quantization scales and zero points, or large sub-32-bit Constants keep the hash and are
covered by accuracy tests. A change to a hashed integral or boolean Constant value, such as
a reshape target or gather index, fails the gate. Any in-scope change, such as one Convert, a rank or a dynamic dimension,
fails the gate, because such changes are exactly what moves FullyConnected, dynamic
quantization or KV-cache selection in a plugin. Changing the scope requires regenerating
all references in the same PR.

Header import preserves typed scalar/array and IEEE nonfinite metadata. Tokenizer
tables/scores/merges/templates and informational labels are explicitly listed as omissions.
`general.tags` is retained for subtype selection; older recipes that list it in
`omitted_metadata` must be re-imported if their tag-dependent conversion becomes relevant.
Record a publisher repository, immutable revision and filename whenever available.
When a publisher commit is unavailable, `revision` is omitted and the case's
`header_sha256` alone pins the header, **not the original weight payload**.
Never infer formats from filenames or replace a mixed-format primary profile with F32.

## Check and update

Build `ov_gguf_frontend_tests` and `openvino_gguf_frontend` in a shared-library test
build (`-DENABLE_TESTS=ON`). The CLI needs Python 3.9+ and the standard library only.
Point `OV_GGUF_GRAPH_RUNNER` (or `--runner`) at the test binary; from the repository root,
the default Linux x86_64 Release output is below (with the
[testing.md](../../docs/testing.md) build, use `build-gguf/bin`):

```sh
export OV_GGUF_GRAPH_RUNNER="$PWD/bin/intel64/Release/ov_gguf_frontend_tests"

python3 src/frontends/gguf/tests/graph_fingerprint.py --report /tmp/changed_cases.json

# Replace CASE_A,CASE_B with the changed case IDs printed by the check.
python3 src/frontends/gguf/tests/graph_fingerprint.py \
  --cases CASE_A,CASE_B --update --report /tmp/changed_cases.json
```

Default mode checks references; CI never updates. On a mismatch, the check prints grouped
representative deltas, how to set the runner, the exact `--update` command for the changed cases,
and a link to the review policy below. In GitHub Actions it also adds an error annotation and a job summary table. Updates require explicit IDs, capture
all selected cases before writing, stage replacements, and roll back a publication failure.
If rollback fails, the error lists retained recovery backups. Existing malformed,
inconsistent or oversized references of the current version must be repaired/reverted
before updating; only references of an older reference version are replaced as new.
Missing/invalid inputs, orphan recipes/references and empty selections fail.
Every invocation writes `changed_cases.json` (in the current directory unless `--report`
is supplied), including partial progress/errors. Keep reports and full graph dumps out of Git.
`--import-gguf <local.gguf> --recipe-output <recipe.json>` reads only the header/tensor
table. For a downloaded header prefix, pass the original file size with `--source-size`
so packed-format detection sees the real final tensor span. It prints the header SHA-256
for the case's `source.header_sha256`, which is stored only in the catalog. References
carry no recipe digest: a recipe edit that changes conversion changes `graph_sha256`.
No checkpoint downloads occur in routine tests.

Comparison is exact: no tolerance, automatic refresh or acceptance of fewer operations.
Deltas identify each changed model-map key, including added and removed models, with
net counts by operation type/version and projection signature, not aligned
individual nodes. A changed hash with equal counts reports a structure/attribute change.
Inspect a full actual graph using the runner's `--gguf-snapshot ... --output ...` diagnostic.

The tools and JSON data are installed beside the C++ test executable. The GGUF Models
precommit CI job checks them for pull requests that directly change the GGUF frontend
component, using Python and the same-run OpenVINO package and test artifacts. It uploads
`changed_cases.json` alongside the model test reports, including on failure. The catalog
check itself needs no source checkout, Python OpenVINO bindings, network access or model
files. It does not run in nightly or per-OS C++ unit-test jobs. The same references are
intended for x86 and ARM; ARM determinism has not yet been validated.

## Review policy for reference changes

A changed reference does not mean every affected checkpoint must be benchmarked in every
scenario. The PR author proves "no regression" on a minimal representative set, chosen
from `changed_cases.json` by the rules below, and widens the set only when a
representative regresses. Initial reference registration establishes a baseline; it is
not evidence of a speedup.

### 1. Group affected cases by change signature

The check groups changes automatically. Each named graph of a case (adaptation variant such
as `ids/language`, `embeds/language` or `vision_embeddings`) is grouped on its own; a
projector identity change is a separate member. A group contains graphs with the same status
that gained or lost the same operation and projection types. Delta magnitudes are ignored
because they scale with depth. The log prints each group once, including its architectures,
members and the full deltas of one representative graph. It prefers a real checkpoint over
synthetic fixtures. `changed_cases.json` keeps every case's deltas and adds the groups under
`change_groups`. Matching signatures do not prove that graph changes are equivalent. Graphs
with a changed hash but equal counts are grouped by graph name only.
Split such groups by the operation types whose structure or attributes changed, using the
runner's `--gguf-snapshot` diagnostic. Describe every group in the PR, identifying the
affected named graphs. Changes to component-map membership must also be explained;
auxiliary encoders and embedding graphs are not exempt.

### 2. Select representatives

Within each group, cover every bucket the changed operations/projections actually touch:

| Axis | Buckets |
| --- | --- |
| Weight format of the changed projections | Q4_0/Q4_1, Q4_K/Q5_K, Q6_K, Q8_0, F16/BF16, low-bit (Q1/Q2/TQ) |
| Topology | dense, sliding-window (Gemma), MoE, hybrid/recurrent (LFM2, Mamba), projector (vision/audio) |
| Size | the smallest real checkpoint in the bucket, plus one model of 7B or more per group |

Use real checkpoints listed in the cases' `models`. Map each `representative` case to the real checkpoints that share its decision class. If fewer than
about three checkpoints are affected in total, benchmark all of them. A change affecting
the whole catalog typically reduces to 8-10 models (for example Llama Q4_K_M, Qwen3 Q4_0,
a Q6_K-heavy model, Qwen3-MoE, LFM2, Gemma-3, a BF16 model, one vision projector and one
model of 7B or more).

### 3. Select devices

- CPU: always.
- GPU: when the change touches MatMul/FullyConnected, quantization or decompression, MoE,
  attention/KV cache, ranks or dynamic dimensions.
- NPU: only for cases explicitly supported on NPU.

Baseline and candidate run on the same machine; name the CPU and GPU hosts used.

### 4. Select scenarios

Every representative runs the baseline scenario: GenAI `LLMPipeline` with
`GGUF_READER=FRONTEND`, batch 1, a prompt of about 1k tokens and 128 generated tokens.
Record prefill latency (time to first token), decode throughput (tokens/s), compile time
and peak memory. Add scenarios only when the change signature triggers them:

| Change touches | Extra scenario |
| --- | --- |
| Attention, RoPE, KV cache, states, PagedAttention adaptation | 8k-token context and continuous batching with 4 parallel requests |
| MoE routing or expert matmuls | batch of 4 or more different prompts |
| Projector graph | one image or audio request |
| Speculative/EAGLE graphs | the speculative pipeline |
| Weight layout or decompression only | baseline scenario only; compile time and peak memory are the main signals |

### 5. Measure, judge and widen

- Alternate baseline and candidate runs (A/B/A/B) with 2 warmups and 5 measured runs;
  report the median and spread.
- A representative passes when no metric is worse by more than the larger of 3% and the
  measured run-to-run noise for that model and device.
- If a representative fails, benchmark every checkpoint in its group. If any group fails,
  also benchmark the second size bucket of the other groups.

### 6. Report

The report format is open; what matters is that reviewers can read the evidence directly
in the PR description. A Markdown table is preferred, for example:

| Model | Device | Scenario | Prefill, ms | Decode, tok/s | Compile, s | Peak memory, GB | Change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B Q4_0 | GPU (Arc A770) | baseline | before -> after | before -> after | before -> after | before -> after | +x% / within noise |

Next to the results, state the grouping and representative mapping from steps 1-2, the
baseline and candidate commits, the hosts and the settings used (precision, plugin,
GenAI and KV options). Raw logs may be linked but are not required in the description.
Structural equality, synthetic timings or unrelated numerical tests do not substitute
for real-model results. Whether a regression is acceptable remains a reviewer decision.

Adding support requires registering every relevant original-dimensional variant and format,
recording provenance and applicable input modes, capturing its reference and reviewing it.
Equivalent descriptor profiles may share a recipe only if all checkpoint identities remain
visible. A model hub list entry without a matching catalog identity fails the Python tests.
