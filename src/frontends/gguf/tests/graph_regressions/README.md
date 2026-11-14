# GGUF structural regression references

These are structural alarms, not runtime benchmarks or numerical equivalence tests.
The native foundation works offline; adapted GenAI model-map capture is blocked on
the frontend adaptation API. Do not describe the deferred stage as final GenAI IR.

## Coverage status

The catalog contains original-dimensional profiles for all 40 native architecture
handlers, including the separate LFM2.5 vocoder, and all 31 projector contracts.
Tiny numerical fixtures do not substitute for original-dimensional profiles.
`GGUFGraphRegression.CatalogCoversEveryNativeHandler` and
`CatalogClassifiesEveryProjector` fail when a registered handler or projector
(plus the composite LFM3 audio-generation branch) has no positive case.

Every checkpoint in the model hub lists (`tests/model_hub_tests/gguf/gguf_{models,mmproj}_*`
and `bonsai_models.json`) appears in a case's `models` as `<repository>/<filename>`.
That entry is added only after the publisher's header, read at an immutable revision,
matches the case's `header_sha256`; filenames alone are never trusted.
`test_catalog_maps_every_documented_checkpoint` enforces the mapping.
Cross-architecture determinism remains an acceptance requirement.

## Artifacts and stages

All paths below are under `tests/test_data/graph_regressions`, in ordinary Git text,
not LFS:

- `catalog.json`: case/checkpoint identities, provenance, stages and projector contracts.
- `schema.json`: shared versioned catalog, recipe and reference definitions.
- `recipes/<case>.json`: compact recipe version 2, one metadata entry or tensor group per
  line. Metadata maps each key to `[type, value]` (`"u32"`, `"f32[]"`, ...); arrays of at
  least 64 values may use `{"rle": [[value, count], ...]}` or, for signed ±1 arrays,
  bit-packed `{"signs": <base64>, "count": n}`. Tensors are `[name, ggml_type, shape]` or
  `[template, ggml_type, shape, layers]`, where one `{}` in the template expands over
  ascending, disjoint layer ranges such as `"0-2,4-63"`. Shapes use OpenVINO order,
  reversing GGUF's `ne[]`; source storage is derived from the GGML type. The runner
  expands this to a per-tensor descriptor before conversion.
- `references/<case>.json`: complete operation type/version counts, projection summaries
  and one `graph_sha256` per captured model. Histogram rows are kept one per line for readable
  diffs; other value arrays are inline. Each reference is limited to **32 KiB across all stages**.

`deferred_plan` uses descriptors that throw on payload access. It reuses production
builder selection and weight allocation/layout planning without allocating checkpoint
weights. This is the all-model precommit candidate, not final materialized IR.
`materialized` runs bounded synthetic GGUF files through ordinary production loading.
`controlled--llama-q8-nonzero-scale` materializes patterned Q8_0 payloads. The 31
legacy positive architecture fingerprints have materialized references with identical
top-level operation and input counts; rejection, broken-model and dependency tests remain.
Numerical repacking, Hadamard and ternary tests remain in the C++ suite.

Synthetic payload generation is versioned by expanded descriptor/tool version 1, uses
fixed finite patterns or zeros, and is limited to 64 MiB per recipe. F32/F16 and patterned Q8_0
controls are supplemented by valid packed-zero blocks for the other supported GGML
formats; each materialization class has a small ordinary-conversion reference.
It never invents random checkpoint-sized payloads. Temporary files are cleaned up.
Nonzero quantization arithmetic is covered by the existing numerical tests;
unrepresented value-dependent final-materialization paths can still escape this gate.

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
| Integral Constant values with at most 64 elements or at least 32-bit integers, and boolean Constants | Shape symbols |
| Parameters, Results and Sinks in model order, with their tensor names | Tokenizer metadata and `fused_names`, `original_precision`, `WeightlessCacheAttribute` |
| ReadValue/Assign variable IDs, types and shapes | |
| GGUF weight descriptors: source tensor, GGML type, storage type/shape, group size, unpacked weight/scale/zero-point layouts and row/column plans | |
| GGUF runtime keys (`gguf*`) on nodes and the model | |
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
Record a publisher repository, immutable revision and filename whenever available.
When a publisher commit is unavailable, `revision` is omitted and the case's
`header_sha256` alone pins the header, **not the original weight payload**.
Never infer formats from filenames or replace a mixed-format primary profile with F32.

## Check and update

Build `ov_gguf_frontend_tests` and `openvino_gguf_frontend` in a shared-library test
build. The CLI needs Python 3.9+ and the standard library only:

```sh
python3 src/frontends/gguf/tests/graph_fingerprint.py \
  --runner build-gguf/bin/ov_gguf_frontend_tests --report /tmp/changed_cases.json

python3 src/frontends/gguf/tests/graph_fingerprint.py \
  --runner build-gguf/bin/ov_gguf_frontend_tests \
  --cases <case-id>,<case-id> --update --report /tmp/changed_cases.json
```

Default mode checks references; CI never updates. Updates require explicit IDs, capture
all selected stages before writing, stage replacements, and roll back a publication failure.
If rollback fails, the error lists retained recovery backups. Existing malformed,
inconsistent or oversized references must be repaired/reverted before updating; update
mode does not silently replace invalid baselines.
Missing/invalid inputs, missing stages, orphan recipes/references and empty selections fail.
Every invocation writes `changed_cases.json` (in the current directory unless `--report`
is supplied), including partial progress/errors. Keep reports and full graph dumps out of Git.
`--import-gguf <local.gguf> --recipe-output <recipe.json>` reads only the header/tensor
table. For a downloaded header prefix, pass the original file size with `--source-size`
so packed-format detection sees the real final tensor span. It prints the header SHA-256
for the case's `source.header_sha256`, which is stored only in the catalog. References
carry no recipe digest: a recipe edit that changes conversion changes `graph_sha256`.
No checkpoint downloads occur in routine tests.

Comparison is exact: no tolerance, automatic refresh or acceptance of fewer operations.
Deltas are net counts by operation type/version and projection signature, not aligned
individual nodes. A changed hash with equal counts reports a structure/attribute change.
Inspect a full actual graph using the runner's `--gguf-snapshot ... --output ...` diagnostic.

The tools and JSON data are installed beside the C++ test executable, so the existing
GGUF-gated C++ CI job can check them without a source checkout, Python OpenVINO bindings,
network access or model files. The same references are intended for x86 and ARM;
ARM determinism has not yet been validated.

## Review policy for reference changes

The PR author must attach baseline-versus-candidate **real-model performance results
for every affected checkpoint identity** in `changed_cases.json`. Include immutable
model/build revisions, hardware/software, precision and plugin/GenAI/KV settings,
inputs, warmups/repetitions, prefill/decode latency or throughput, compile time and
peak memory as applicable. Preserve raw results and explain structural changes.
Do not substitute structural equality, synthetic timings or unrelated numerical tests
for those results.

For synthetic/controlled references, identify the affected quantization/topology decision
classes and map them to real checkpoints. Reviewers may approve benchmark representatives
only with an explicit mapping and justification for **each** affected model. There is no
automatic equivalence exemption, regression threshold waiver or benchmark-result validator.
Initial reference registration establishes a baseline; it is not evidence of a speedup.

Adding support requires registering every relevant original-dimensional variant and format,
recording provenance, capturing the required stages and reviewing the new references.
Equivalent descriptor profiles may share a recipe only if all checkpoint identities remain
visible. A model hub list entry without a matching catalog identity fails the Python tests.
