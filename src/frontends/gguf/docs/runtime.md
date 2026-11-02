# Running converted GGUF models

`convert()` returns a stateless graph with llama.cpp-style inputs. A caller-registered `GenAIExtension`
turns a language model into a stateful model with the OpenVINO GenAI interface. Multimodal projector
files use their own [encoder contract](mmproj.md#graph-boundary).

## Split GGUF files

The native frontend loads split models directly without writing a merged GGUF. Pass any shard
with the standard `model-00001-of-00003.gguf` filename; the loader reads `split.count` and
`split.no` and opens the other expected shards in the same directory. All shards of the
chosen quantization must be available locally.

```python
from openvino.frontend import FrontEndManager

frontend = FrontEndManager().load_by_framework("gguf")
model = frontend.convert(frontend.load(
    "Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"
))
```

For renamed shards or files in different directories, supply a complete list of string paths.
List order does not matter; the loader uses `split.no` to select shard zero's model metadata.

```python
model = frontend.convert(frontend.load([
    "/models/part-a.gguf", "/other-disk/part-b.gguf", "/models/part-c.gguf"
]))
```

C++ accepts a `std::vector<std::string>` or `std::vector<std::filesystem::path>`, and also
separate path arguments to `load()`:

```cpp
auto input = frontend->load(std::vector<std::string>{
    "/models/part-a.gguf", "/other-disk/part-b.gguf", "/models/part-c.gguf"});
auto model = frontend->convert(input);
```

An explicit list must contain every shard; the loader does not fill in omitted entries.
Inputs must belong to one split model. Missing shards,
duplicate shard indices or tensor names, conflicting metadata and inconsistent total tensor
counts produce load errors. Each shard keeps its own mapping; quantized weights use the same
repacking as single-file inputs. This avoids a merged disk copy but does not reduce conversion
or compilation memory requirements or extend architecture and quantization coverage.

## Stateless contract

The default decoder graph mirrors the llama.cpp backend interface:

| Input | Meaning |
|---|---|
| `inp_tokens [1,1,1,T]` i32 | Token IDs of this step |
| `inp_pos [1,1,1,T]` i32 | Positions; four sections per token for interleaved M-RoPE (`qwen35`, `qwen35moe`) |
| `inp_out_ids [1,1,1,N]` i32 | Rows whose logits are produced |
| `self_kq_mask [1,1,T,KV]` f32, `self_kq_mask_swa` | Additive causal mask; the second one for sliding-window layers |
| `inp_kv_idx`, `token_len_per_seq` | KV write slots and per-sequence length, when the graph consumes them |
| `cache_k_*`, `cache_v_*`, recurrent states | Previous cache/state, returned as matching outputs |

The output is `logits`. [`run_gguf_model`](../../../../tests/model_hub_tests/gguf/utils.py) is a
working Python driver for two steps. This is the only route available from Python: the GGUF passes
below have no Python bindings, and `DecoderTransformationExtension` cannot be constructed from Python.

## Stateful and GenAI conversion

Register `GenAIExtension` before `convert()`. The frontend creates stateful caches before
stateless cache lowering, completes normalization (including recurrent-operation fusion), then
adapts the normalized graph to GenAI. Callers do not register or order individual passes.

```cpp
#include <openvino/frontend/gguf/extension/genai.hpp>
#include <openvino/frontend/manager.hpp>

using namespace ov::frontend::gguf;
ov::frontend::FrontEndManager manager;
auto frontend = manager.load_by_framework("gguf");
frontend->add_extension(std::make_shared<GenAIExtension>());
auto model = frontend->convert(frontend->load("model.gguf"));
```

Internally, [`GGUFMakeStateful`](../include/openvino/frontend/gguf/make_stateful.hpp) replaces each KV cache
Parameter/Result pair with a Variable, adds `beam_idx` and reorders caches by it. Sliding-window
caches can be left stateless with `skip_caches`. Recurrent states are declared through the
`gguf_recurrent_states` rt_info key.

[`AdaptToGenAI`](../include/openvino/frontend/gguf/adapt_to_genai.hpp) requires a stateful model and
exposes GenAI's interface. Select the mode in the `GenAIExtension` constructor:

| Mode | Inputs | Output |
|---|---|---|
| `IDS_TO_LOGITS` (default) | `input_ids [B,T]`, `attention_mask [B,KV]`, `position_ids [B,T]`, `beam_idx [B]` | `logits [B,T,vocab]` |
| `EMBEDS_TO_LOGITS` | `inputs_embeds [B,T,D]` instead of `input_ids`; see [multimodal language models](mmproj.md#using-the-encoders-and-a-language-model) | `logits [B,T,vocab]` |

- The leading dimensions are derived from the live inputs, so one graph serves both SDPA and
  `ov::pass::SDPAToPagedAttention`.
- The LM head is connected directly to the logits Result with a rank-3 input, so GenAI's
  slice-before-matmul optimization projects only the sampled rows during prefill.
- Sliding-window masks use the `gguf_swa_window_size` rt_info value when the file records a window length.
- Gated-DeltaNet states (`qwen35`, `qwen35moe`) and LFM2 short-convolution states
  (`lfm2`, `lfm2moe`) get a dynamic batch, follow `beam_idx` and mask
  left padding. Mamba 2 and `nemotron_h` states stay at one sequence: greedy, batch-1 decoding only.
  A consumer must reset, not trim, the states listed in `gguf_recurrent_states`.
- The pass does not create a tokenizer, sampler or pipeline, and it does not adapt encoder models.

`GGUFMakeStateful` remains independently available through `DecoderTransformationExtension`
for consumers that keep GGUF IO. The llama.cpp backend uses its own stateful lowering for its
cache-slot and fixed-mask semantics, without GenAI adaptation.

## Tokenizer metadata

Conversion attaches the file's `tokenizer.ggml.*` and `tokenizer.chat_template` values to the
model's rt_info as [`GGUFTokenizerMetadata`](../include/openvino/frontend/gguf/tokenizer_metadata.hpp),
under `gguf_tokenizer_metadata_key()`, when the decoder provides them. Values are strings, string vectors or tensors. A consumer such
as GenAI can build the tokenizer from it without reopening the file. The attribute is intentionally
in-memory only: it is dropped on `clone()` and not written to IR, so read it from the converted model.

## Internal operations and serialization

[`SetRows`](../include/openvino/frontend/gguf/set_rows_op.hpp) is a conversion placeholder for
`GGML_OP_SET_ROWS`. `GGUFMakeStateful` consumes cache writes; `LowerSetRowsStateless` lowers the rest.
It never survives into the converted model.

These core `ov::op::internal` operations can remain in the converted model:

| Operation | Emitted for | Alternative |
|---|---|---|
| `SelectiveSSM` | `SSM_SCAN`, Mamba 2 scalar decay | None |
| `GatedDeltaNet` | `GATED_DELTA_NET`, scalar gate | `Loop` reference path for per-key gating or `force_ref`; not available with `split_outputs` |
| `GatherMatmul` | `MUL_MAT_ID`, eligible constant expert weights | Generic Gather/MatMul for other weights |
| `FullyConnectedCompressed` | Experimental raw GGUF matrix weights | Default decomposed weight path |
| `GatherCompressed` | Experimental raw GGUF embedding weights in the six supported formats | Default decomposed weight path |

See [`ssm_scan.cpp`](../src/op/ssm_scan.cpp), [`gated_delta_net.cpp`](../src/op/gated_delta_net.cpp)
and [`mul_mat_id.cpp`](../src/op/mul_mat_id.cpp) for exact selection conditions.

Internal operations are outside the public IR opsets. Saving IR can succeed and still produce a file
the IR frontend cannot read back, so test the save/read round trip. The IR reader explicitly restores
`FullyConnectedCompressed` and `GatherCompressed` from `ie_internal_opset`, including their raw GGUF
Constants, using block-aware byte counts. Without external-reference metadata, ordinary XML and BIN retain the raw payload; this does not
automatically create external GGUF references or preserve the original file mapping. File regressions
check payload identity, shared weights and GPU results after the original mapping is released.

Explicit external references are supported for unchanged raw Constants carrying all three internal
`gguf_ext_source`, `gguf_ext_offset` and `gguf_ext_size` metadata fields. The serializer writes the bare
GGUF filename, byte offset and exact block-aware payload size into the Const data attributes instead
of copying that payload to BIN. Load the paired XML/BIN through the model-file entrance: the reader
resolves the source relative to the weights-file directory and checks the filename, GGUF magic,
type/shape/size and overflow-safe file range. In-memory XML has no such directory and is rejected.
The current entrance also requires the accompanying BIN path; external-only XML without BIN is not
covered. The target FileWeightsProvider shares providers/regions within a read and returns owned,
copy-backed buffers with existing descriptors, not mx's zero-copy mmap. Its no-mmap guard is unchanged.
Loaded payloads remain snapshots if the source is subsequently replaced; a fresh read observes the
replacement. This is not a content-integrity check or a symlink-confinement guarantee.

Reading an external reference restores its three metadata fields and an internal resolved
`gguf_ext_source_path`. Before another external export, the serializer checks that resolved file's
identity, magic, range and complete payload against the loaded Constant snapshot. Missing, truncated
or changed sources raise an error instead of silently exporting a stale reference. Explicitly remove
all four fields to materialize the snapshot into an ordinary BIN. Manually supplied three-field
metadata without a resolved path remains caller-explicit and does not receive this snapshot check.
File regressions cover successive same-directory exports/read-backs, stale-source rejection and
explicit BIN materialization. This does not guarantee relocation to another directory or atomic
integrity if a source changes after validation; snapshot comparison also reads the complete payload.

The native file loader records source information for explicitly preserved raw GGUF tensors. The
shared weight converter automatically annotates a Constant only while its original Tensor pointer,
type, shape and complete byte size still match. Unchanged tied aliases retain the source; copied
or row-sliced allocations do not inherit it. Compatibility unpack/requantization and supplied
decoders without parser source records remain unchanged. SG repacking clears all four source
fields on its new Constant while leaving the original raw Constant intact. These checks are not a
general provenance guarantee for arbitrary transforms or mutable aliases.

Source paths use UTF-8 metadata and native filesystem paths, including external IR read-back.
Shards with the same bare filename disable automatic references for those sources, so they can
still be saved into ordinary BIN without ambiguous sibling references. Otherwise place the source
GGUF files beside the exported BIN before loading it; arbitrary destination relocation is not
automatically resolved. File regressions cover automatic sources in eleven raw formats, shared
aliases, copied/sliced payloads, and a Unicode Q4_K source through external IR read-back/re-export.

During one GPU graph build, descriptor-backed
raw GGUF Constants can reuse a previously uploaded primitive despite different CPU pointers. The
source index includes descriptor identity/offset, byte size, shape, type and batch interpretation;
candidate payloads must match byte-for-byte, so source-ID collisions or changed snapshots cannot
merge different data. Only full uploads enter this index; SG-marked Constants and builds using a
cross-device weight-sharing context bypass it. Existing pointer caching and shared-buffer ownership
remain in place. This is not a cache across compiled models or processes. Broader transformation
provenance and complete-model external IR remain separate acceptance work. Other internal
operations in the table do not gain IR support from this path. Compiled-model export, import and
caching are plugin-specific; test them on the target device. In-process conversion needs no IR.

GPU GGUF implementations serialize their execution-selection state with the compiled kernels:
OCL shuffle flags, prefill threshold and F16/DQ selection, and CM weight format and threshold.
Import restores these values rather than using the import process's environment defaults;
temporary transcode and activation scratch buffers remain lazily allocated. This fixes missing
derived state in the source implementation as well, without changing kernels or default routing.
No compatibility with blobs exported before this state was serialized is asserted.

File regressions export and import eleven raw FC formats through a fresh Core after external IR
read-back, checking dynamic rows, new input buffers, original numerical limits and executed stages.
The six applicable formats also exercise actual CM after import. A Q4_0 regression separately
checks same-process, fresh-Core cache misses/hits for static and dynamic raw/SG graphs, payload-tail
invalidation, and two independent asynchronous requests. These are full compiled blobs, not
weightless or encrypted caches. Cross-process caching, fused signatures, cancel/reset, explicit
multi-stream execution and full-checkpoint serialization remain separate acceptance work.

A new internal-op path should keep a public-op reference path where practical, test both against
the same oracle and update the table above. Frontend placeholders need a normalization lowering.
