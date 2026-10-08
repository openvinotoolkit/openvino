# Debugging GGUF accuracy

First identify the conversion path and failing boundary. Use the cheapest relevant comparison;
there is no requirement to complete every diagnostic before inspecting the implicated code.

| Execution path | Start with | Compare |
|---|---|---|
| Native decoder | [Architecture fixtures](../tests/test_data/arch_accuracy/README.md) | Full logits on identical histories: prefill, one-token decode, then cache append |
| Native mmproj | [Encoder fixtures](../tests/test_data/mmproj_accuracy/README.md) | Same preprocessed pixels/features and indices, each modality, raw and adapted embeddings |
| llama.cpp OpenVINO backend | [Backend isolation](#llamacpp-backend-isolation) below | Genuine ggml CPU execution versus the OpenVINO backend |

Native builders and supplied cgraph decoders share converters, but have different graph construction
and state/IO adaptation. Backend environment variables do not configure native frontend inference.
See [testing](testing.md) for commands, acceptance criteria, and how to recognize skipped coverage.

## Reference and precision

Use the real llama.cpp/ggml CPU implementation as the reference for model outputs and
layout-sensitive operations. A hand-written recurrence or NumPy layout transformation can repeat
the implementation's misunderstanding. Simple elementwise checks with unambiguous closed forms
may compute their expectations inline; see [operation tests](how_to_add_op.md#test-and-the-coverage-gate).

Record both revisions, checkpoint and quantization, input tensors or token history, device,
inference precision, activation quantization, and cache precision. Use reference-token replay
after a greedy choice differs; comparing later logits on different histories hides the cause.
Coherent text is a functional check, not numerical acceptance.

The same checkpoint does not imply the same arithmetic. Compare against both native quantized
CPU execution and, when needed, F32 arithmetic on the checkpoint's represented weights. Expanding
weights preserves the file's original quantization error; it does not recover the publisher's
F32 weights. Keep these measurements separate and retain failed comparisons. See
[quantization](quantization.md) for conversion losses, native precision controls, and memory costs.

## Localize the failing boundary

1. **Inputs:** align tokenization, prompts, chat templates, masks, positions, and preprocessing.
   For mmproj, check temporal pairs, patch/window order, image range, audio frame layout, and
   feature packing against [the input contracts](mmproj.md#graph-boundary).
2. **Time or shape:** compare prefill with decode, then several sequence/image sizes on the
   same request. Exercise nonzero positions, window boundaries, and state reset. A decode-only
   failure suggests cache updates or shape-dependent indexing, but does not prove their guilt.
3. **Adaptation:** compare native outputs before and after `GGUFMakeStateful`, `AdaptToGenAI`,
   or `AdaptMmprojToGenAI` as appropriate. Use independent clones/requests and retain the same
   logical inputs. For combined mmproj files, isolate vision and audio separately.
4. **Intermediates:** compare corresponding tensors in element order using NMSE, absolute/relative
   error, and cosine where norms are nonzero. Inspect the first material divergence, then test
   whether correcting or bypassing it fixes the final output.
5. **Regression:** reproduce the implicated operation at realistic dimensions and keep an oracle
   fixture. Test all affected converter paths and rerun shared architecture/projector regressions.

A cosine cliff localizes a discrepancy; it does not establish a permutation bug. A smooth slope
can suggest accumulated numerical error but is not sufficient to exonerate structure. Sums,
means, minima, and maxima cannot detect permutations. Equal sorted buffers with different element
order support a layout hypothesis; unequal sorted buffers can also mean the wrong source region
was selected. Check shapes, strides, offsets, and candidate source tensors before blaming arithmetic.

Near an MoE boundary, compare router logits, selected expert IDs, and the selection cutoff margin.
A small change near a tie can select another expert and amplify later differences. Identifying
this sensitivity does not waive the agreed accuracy threshold.

## llama.cpp backend isolation

Use a compatible backend revision; the repository/ref used by CI are recorded in
[the backend workflow](../../../../.github/workflows/job_gguf_llamacpp_validation.yml).
For a model comparison, build ggml CPU and OpenVINO variants from the same llama.cpp revision.
`GGML_OPENVINO_DEVICE=CPU` selects OpenVINO on CPU, not the ggml CPU oracle. Do not assume
`--device none` or unsetting an unrelated variable bypasses a compiled-in backend.

From that llama.cpp checkout, create an independent CPU reference build:

```sh
cmake -S . -B build-cpu -DGGML_OPENVINO=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build-cpu --target llama-simple llama-eval-callback --parallel
./build-cpu/bin/llama-simple -m model.gguf -n 12 "The capital of France is"
./build-ov/bin/llama-simple -m model.gguf -n 12 "The capital of France is"
```

Here `build-ov` is the separately configured OpenVINO build. Raw completion avoids chat-template
differences during bisection; test chat behavior separately with matched templates. If both runs
fail, check the reference/model setup before attributing the problem to OpenVINO. If both produce
coherent text, continue with the required numerical comparisons.

### Capturing intermediates

`llama-eval-callback` captures ggml node outputs during execution. Its printed sums are useful
for a preliminary scan; obtain full tensors for an elementwise comparison. Match node identity,
shape, and execution step: names may repeat, and positional alignment ends when graphs diverge.

Backend instrumentation lives in `ggml/src/ggml-openvino/`, with CPU callback instrumentation in
`common/debug.cpp`. Check the selected checkout for each control before using it; these are
backend implementation details, not frontend API guarantees.

| Control | Purpose |
|---|---|
| `GGML_OPENVINO_STATEFUL_EXECUTION=1` | Exercise backend KV/recurrent state handling |
| `GGML_OPENVINO_DUMP_CGRAPH=1` | Inspect subgraph operations and tensor geometry |
| `GGML_OPENVINO_DEBUG_INPUT=1` / `GGML_OPENVINO_DEBUG_OUTPUT=1` | Inspect bound subgraph inputs/outputs and summary statistics |
| `GGML_OPENVINO_DUMP_TENSOR=<substring>` | Capture matching OV outputs in element order |
| `GGML_DUMP_TENSOR=<substring>` | Capture matching CPU eval-callback tensors |
| `GGML_OPENVINO_DUMP_IR=1` | Inspect serialized graphs, subject to [internal-op limits](runtime.md#internal-operations-and-serialization) |
| `GGML_OPENVINO_FORCE_F32=1` | Investigate backend precision differences |
| `GGML_OPENVINO_DISABLE_TYPES=Q8_0,Q6_K` | Isolate handling of selected quantized types |
| `GGML_OPENVINO_DISABLE_OPS=SSM_CONV,DIV` | Investigate selected operation families through backend fallback |

Capture tensors while their buffers are valid. ggml scratch memory can be reused after an
operation, so a later read of an interior tensor can be stale. Genuine boundary outputs and
appropriately captured caches are useful comparison points. Promoting extra outputs or enabling
callbacks can change allocation and graph partitioning; a new crash under instrumentation is
inconclusive. Matching a cache write bounds only the compared branch, not every parallel path.

Fallback experiments can change graph partitioning, precision, and boundary handling. Correct
output after disabling an operation implicates that path or its integration, not necessarily
the kernel itself. Unchanged output does not exonerate it if another error masks its effect.
Enumerate the types actually present before attempting to isolate quantization; a fixed list
of formats is not proof that every quantized operation has been bypassed.

## The ggml-CPU oracle

Build a small program linking the same ggml CPU libraries as the reference. Construct a graph,
fill asymmetric inputs, run it, and save all outputs. Existing examples include
[`gdn_oracle.c`](../tests/gdn_oracle.c), [`ssm_scan_oracle.c`](../tests/ssm_scan_oracle.c), and
[`imrope_oracle.c`](../tests/imrope_oracle.c). With `LLAMA_SRC` and `LLAMA_BUILD` pointing to the
CPU-only checkout and build:

```sh
cc op_oracle.c -o op_oracle -I "$LLAMA_SRC/ggml/include" \
  -L "$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" -lggml -lggml-base -lggml-cpu -lm
./op_oracle
```

Read the operation's dimension contract in `ggml.h`; use `no_alloc = true` with
`ggml_backend_alloc_ctx_tensors`. Include multiple heads/tokens and unequal query/KV dimensions
where relevant. Assert fused and decomposed converter paths against the oracle. Model and encoder
oracles have separate [regeneration instructions](testing.md#fixtures-and-oracles).

## Recurring failure patterns

| Pattern | What to inspect |
|---|---|
| Partial rotary geometry | Read rotary width/offset from the operation configuration; do not rotate a whole head when only part is rotary |
| Shared RoPE tables | The table and each consumer must agree on mode, dimensions, and position sections |
| Dynamic tail slices | Derive offsets from the live length; a prefill offset can overrun during decode |
| Strided VIEW treated as dense | Inspect `ne[]`, `nb[]`, element offset, and semantic `op_case`; shape alone does not establish contiguity |
| Recurrent state shared across subgraphs | Keep per-subgraph bookkeeping separate and test beyond the first decode step |
| Media adaptation mismatch | Check modality pruning, removed input prefixes, DeepStack splitting, and separate Gemma4 per-layer embeddings |

Use small fixtures while localizing. Compilation may materialize additional weight buffers;
reducing context length does not resolve weight-related memory pressure. Record peak memory for
large checkpoints. Redirect long-running command output to a file; retain the launched PID and
terminate that process if necessary, rather than killing every process matching a model runner name.
