# QSA and QSASDPA Specification for OpenVINO

**Status:** Design-for-review / Intel GPU plugin implementation in progress

**Audience:** OpenVINO operator and plugin maintainers, graph-transformation reviewers, kernel and performance reviewers

**Scope:** Qwen Sparse Attention (QSA), including the paged-cache `QSA` operation, the dense-cache `QSASDPA` operation, the `QSAIndexerCache` state holder, and the associated graph-fusion passes.

## 1. Executive summary

Qwen Sparse Attention (QSA) is a sparse-attention mechanism consisting of an indexer and a selected-block attention stage. The indexer scores compressed key blocks, selects the top-$k$ blocks, and the main attention evaluates only the selected blocks plus the causal tail required for the current position.

This proposal exposes two related GPU operations:

- `ov::intel_gpu::op::QSA`: a paged-cache operation for callers that already manage physical pages and block tables, such as a continuous-batching scheduler.
- `ov::intel_gpu::op::QSASDPA`: an SDPA-style sibling that consumes dense, state-managed cache tensors and hides the physical page table from the model graph.

Both operations share the same `QSA::Config`, numerical semantics, host dispatch logic, and CM kernel families. `QSASDPA` uses the shared QSA implementation in dense mode with runtime-resolved cache layouts; it does not introduce a second attention implementation.

The design deliberately preserves OpenVINO fallback behavior. A graph transformation is allowed to replace a generic graph only when the required topology, shapes, attributes, cache ownership, and rotary-table representation can be proven. Otherwise the original graph remains valid and is left unchanged.

## 2. Goals and non-goals

### 2.1 Goals

1. Provide a fused operation definition for Qwen Sparse Attention(QSA).
2. Define PA-style QSA interface (`ov::intel_gpu::op::QSA`) for callers that manage physical pages and block tables.
3. Define SDPA-style QSA interface (`ov::intel_gpu::op::QSASDPA`) and Indexer Cache (`ov::intel_gpu::op::QSAIndexerCache`) for dense/KVCache-managed deployments.
4. Make sure to reuse the same kernels for both the PA-style and SDPA-style QSA implementations.
5. Preserve causal, ragged-token, GQA/MQA, compressed-indexer, and output-gate semantics.
6. Provide OpenVINO graph transformations pattern pass for both PA-style and SDPA-style QSA operations.
7. QSA indexer's RoPE is not standard RoPE, fuse it into the QSA operation instead of handling it out of QSA operation.
8. Make unsupported cases explicit through validation or conservative graph-pass bail-out.
9. Leave room for dynamic batch, beam reordering, rotary-table repacking, and more efficient dense kernels in later revisions.

### 2.2 Non-goals for version 1

- Define and implement a general-purpose sparse-attention API for all sparsity patterns.
- Modifying or reusing the implementation of `PagedAttention` or standard `ScaledDotProductAttention`.
- Supporting arbitrary dynamic batch size in the dense `QSASDPA` lowering.
- Supporting beam reordering inside QSA kernels.
- Performing graph-side repacking of separate `full_cos` and `full_sin` rotary tables.
- Changing the semantics of the existing paged `QSA` operation.

## 3. Mathematical and semantic definition

Let:

- $D$ be the hidden size used by the indexer projection;
- $H_q$, $H_{kv}$ be the main-attention query and KV head counts;
- $D_h$, $D_v$ be the main-attention key and value head widths;
- $H_i$ be the number of indexer heads;
- $D_i$ be the indexer head width;
- $R$ be the compression ratio, also called `compress_ratio`;
- $K_b$ be the number of selected complete blocks, `block_topk`;
- $B$ be the runtime batch/sequence-slot count.

The indexer computes a query/key representation, applies the configured RMS normalization and partial rotary embedding, and forms block scores:

$$
S_{b,j} = \frac{1}{\sqrt{D_i}}
\sum_{h=0}^{H_i-1}\mathrm{ReLU}
\left(Q_{b,h} K_{j,h}^{T}\right).
$$

The score path is followed by the causal mask and `TopK`. The selected complete blocks are passed to the sparse main-attention stage. The current incomplete block is handled by the causal-tail path and is not incorrectly discarded merely because it is not complete.

The operation does not own the surrounding model projections. Main `q_proj`, `k_proj`, `v_proj`, `o_proj`, gated residual operations, and unrelated post-processing remain ordinary OpenVINO graph nodes unless a separate, independently proven transformation handles them.


The whole QSA operation can be broken down into the following stages:
  - Q0: Append the current projected K/V to the paged main cache 
  - Q1: Build indexer Q/K, apply normalization and RoPE, append raw K, and produce completed-block summaries 
  - Q2: Score visible compressed blocks and select TopK candidates 
  - Q3: Read selected blocks and the causal tail, then evaluate sparse attention


## 4. Operation definitions

### 4.1 `ov::intel_gpu::op::QSA`

`QSA` is the paged-cache form. Its OpenVINO internal base operation is `ov::op::internal::QSA`; the Intel GPU class exposes the same operation under the `gpu_opset` type information.

#### Inputs

`QUERY` has a deliberately narrow meaning in this specification. It is the **prepared
main-attention query** consumed directly by Q3. It is not the raw `MIXED` tensor and it is
not the indexer query generated by Q1. The surrounding graph owns the main `q_proj` and
any model-specific main-attention normalization, rotary embedding, head reshaping, and
dtype conversion. QSA does not repeat those operations. In particular, Q3 only loads
`QUERY`, applies the configured main-attention scale, and uses it for dot products with
the selected main K cache. The indexer query follows a separate path:
`MIXED -> INDEXER_QK_WEIGHT -> Q1 RMSNorm/partial RoPE`.

Without an output gate, `QUERY` is `[tokens, Hq*Dh]`, with head data flattened in the
last dimension. When `has_output_gate` is enabled, it is `[tokens, 2*Hq*Dh]` and contains
the main query and the per-head output-gate payload in the implementation-defined halves;
the gate is consumed by Q3 and is not an additional indexer input. A graph transformation
must prove that the source tensor already has this prepared-query contract before wiring
it to QSA. A raw `q_proj` output may be used only when the model proves that no additional
main-attention Q processing is required.

| Index | Name | Required shape/meaning |
|---:|---|---|
| 0 | `MIXED` | `[tokens, D]`; output of the gated-residual read/mix stage |
| 1 | `QUERY` | Prepared main-attention Q: `[tokens, Hq*Dh]`, or `[tokens, 2*Hq*Dh]` with the output-gate payload. It is consumed directly by Q3; QSA does not apply main-Q RMSNorm or RoPE. |
| 2 | `KEY` | Current token keys, `[tokens, Hkv*Dh]` |
| 3 | `VALUE` | Current token values, `[tokens, Hkv*Dv]` |
| 4 | `PAST_LENS` | `[num_seqs]`; integer history length before the current call for each sequence. It selects the append position and visible causal prefix. |
| 5 | `SUBSEQUENCE_BEGINS` | `[num_seqs+1]`; prefix offsets into ragged token-major `MIXED`, `QUERY`, `KEY`, and `VALUE`. Sequence `s` occupies `[begins[s], begins[s+1])`. |
| 6 | `BLOCK_INDICES` | `[used_blocks]`; physical page IDs. Logical page `floor(position/PA_BLOCK_SIZE)` is translated through the per-sequence range selected by `BLOCK_INDICES_BEGINS`. |
| 7 | `BLOCK_INDICES_BEGINS` | `[num_seqs+1]`; offsets into `BLOCK_INDICES`, one range per sequence. |
| 8 | `KEY_CACHE` | Main K cache, `[num_blocks, Hkv, PA_BLOCK_SIZE, Dh]`, f16 in the current GPU implementation. It is written by Q0 and read by Q3. |
| 9 | `VALUE_CACHE` | Main V cache, `[num_blocks, Hkv, PA_BLOCK_SIZE, Dv]`, f16 in the current GPU implementation. It is written by Q0 and read by Q3. |
| 10 | `INDEXER_RAW_K_CACHE` | Unnormalized/unrotated raw indexer-K state, `[num_blocks, 1, PA_BLOCK_SIZE, Di]`; written by Q1 and used to form summaries. |
| 11 | `INDEXER_SUMMARY_CACHE` | Normalized/rotated completed-block summaries, `[num_blocks, 1, PA_BLOCK_SIZE/R, Di]`; written by Q1 and read by Q2. |
| 12 | `INDEXER_QK_WEIGHT` | Fused indexer projection, `[H_i*D_i + D_i, D]`; the first `H_i*D_i` rows produce indexer Q and the final `D_i` rows produce indexer K. |
| 13 | `INDEXER_Q_NORM_WEIGHT` | Indexer-Q RMSNorm gain, `[D_i]`. |
| 14 | `INDEXER_K_NORM_WEIGHT` | Indexer-K/summary RMSNorm gain, `[D_i]`. |
| 15 | `ROTARY_COS_SIN` | One already-packed table `[max_positions, indexer_rotary_dim]`, indexed by `POSITION_IDS`; separate dynamic `full_cos`/`full_sin` inputs are not this contract. |
| 16 | `POSITION_IDS` | `[tokens]`; absolute token positions or the position rank used by the supported mRoPE representation. Q1 uses it for indexer-query RoPE and uses the block-start position for completed summaries. |

#### Output

The output is the main-attention result. Its token and head dimensions follow the `QUERY`, `Config::heads_num`, `Config::kv_heads_num`, `Config::v_head_size`, and output-gate configuration.

`QSA` does not own persistent state. The caller supplies page tables and cache tensors on every execution. This is intentional: it supports scheduler-controlled physical memory and continuous batching.

#### Cache layouts and ownership

The main KV cache is a physically paged, token-major-within-page layout:

```text
KEY_CACHE   [physical_page, kv_head, token_in_page, Dh]
VALUE_CACHE [physical_page, kv_head, token_in_page, Dv]
```

For logical position `p`, Q0 computes `logical_page = p / PA_BLOCK_SIZE`, looks up the
physical page through `BLOCK_INDICES`, and uses `token_in_page = p % PA_BLOCK_SIZE`.
`BLOCK_INDICES` therefore belongs to the caller/scheduler; QSA does not allocate or
reorder physical pages.

The indexer cache uses the same physical-page indirection but has one indexer KV head:

```text
INDEXER_RAW_K_CACHE [physical_page, 0, token_in_page, Di]
INDEXER_SUMMARY_CACHE [physical_page, 0, summary_in_page, Di]
```

where `summary_in_page = qsa_block_in_page` and one summary represents `R` consecutive
tokens. Raw K is stored before Q1's K normalization/RoPE; summaries are stored after
mean pooling, K normalization, and block-start RoPE. These are distinct semantic layouts,
not interchangeable views of the same buffer.

### 4.2 `ov::intel_gpu::op::QSASDPA`

`QSASDPA` has the same attention and indexer semantics as `QSA`, but removes physical page-table inputs from the public graph contract. It is intended for a graph in which main K/V state is managed like standard SDPA state and indexer state is managed by `QSAIndexerCache`.

#### Inputs

`QUERY` has exactly the same prepared-query contract as in `QSA`: it is the dense SDPA
boundary's main-attention Q after all required external main-Q processing. `QSASDPA` does
not normalize or rotary-embed this tensor. Its indexer Q is still generated internally
from `MIXED` by Q1. This distinction is important because the dense SDPA graph may expose
several Q-like tensors; only the tensor already suitable for the main SDPA Q operand may
be connected to `QSASDPA::QUERY`.

| Index | Name | Required shape/meaning |
|---:|---|---|
| 0 | `MIXED` | `[tokens, D]` |
| 1 | `QUERY` | Prepared main-attention Q: `[tokens, Hq*Dh]`, or `[tokens, 2*Hq*Dh]` when the output gate is carried with it. |
| 2 | `PAST_LENS` | `[num_seqs]`; history length before this call. |
| 3 | `SUBSEQUENCE_BEGINS` | `[num_seqs+1]`; ragged-token offsets. |
| 4 | `KEY_CACHE_DENSE` | Dense main K state, `[B, Hkv, L_k, Dh]`; upstream KV cache owns current-token append. |
| 5 | `VALUE_CACHE_DENSE` | Dense main V state, `[B, Hkv, L_v, Dv]`; upstream KV cache owns current-token append. |
| 6 | `INDEXER_RAW_K_CACHE_DENSE` | Runtime dense raw-K view, `[B, L_raw, D_i]`, supplied by the `qsa_inst` indexer-cache state descriptor. |
| 7 | `INDEXER_SUMMARY_CACHE_DENSE` | Runtime dense summary view, `[B, L_summary, D_i]`, supplied by the `qsa_inst` indexer-cache state descriptor. |
| 8 | `INDEXER_QK_WEIGHT` | Fused indexer projection, `[H_i*D_i + D_i, D]`. |
| 9 | `INDEXER_Q_NORM_WEIGHT` | Indexer-Q RMSNorm gain, `[D_i]`. |
| 10 | `INDEXER_K_NORM_WEIGHT` | Indexer-K/summary RMSNorm gain, `[D_i]`. |
| 11 | `ROTARY_COS_SIN` | Already-packed `[max_positions, indexer_rotary_dim]` table. |
| 12 | `POSITION_IDS` | `[tokens]`; absolute positions or supported mRoPE ranks. |

`KEY` and `VALUE` current-token inputs are intentionally absent: their values have already been incorporated into the dense cache by the upstream cache operation. The QSA lowering therefore uses `skip_kv_write=true`.

#### Dense lowering and runtime addressing

The Intel GPU lowering uses `cldnn::qsa` with `dense_mode=true` and `skip_kv_write=true`.
It retains the existing QSA stage implementation while deriving the effective dense
page/block span from the **current resolved cache layout**, rather than from a fixed
`Config::pa_block_size`. The lowering still supplies identity page metadata
(`BLOCK_INDICES = [0, ..., B-1]` and `BLOCK_INDICES_BEGINS = [0, ..., B]`) because the
existing primitive contract requires these inputs; dense-mode address calculation treats
each sequence as its own contiguous degenerate page and uses the runtime length/stride.
This is an implementation bridge, not a fixed-capacity requirement.

The dense cache layout is a logical, state-managed view:

```text
KEY_CACHE_DENSE   [batch, kv_head, position, Dh]
VALUE_CACHE_DENSE [batch, kv_head, position, Dv]
INDEXER_RAW_K_CACHE_DENSE [batch, position, Di]
INDEXER_SUMMARY_CACHE_DENSE [batch, completed_block, Di]
```

The current implementation passes dense rank-3 raw-K/summary state directly to dense mode;
no fixed-capacity reshape is required. The Q1/Q2/Q3 CM stages are the same validated QSA
stages, with dense-mode runtime addressing and runtime dispatch parameters. Q0 is not
scheduled because the upstream dense `KVCache` has already appended the current K/V. No
second attention implementation is introduced by `QSASDPA`.

#### Shape and batch contract

The following dimensions are resolved at runtime and may change between calls:

- main K/V cache sequence lengths;
- raw-K length;
- summary length;
- current token count and per-sequence history lengths.

The following properties remain required at graph/lowering time:

- cache ranks, head counts, head widths, element types, and dense layout semantics must
  match the QSA configuration;
- the batch/sequence-slot count `B` must be statically known;
- `PAST_LENS` and the sequence metadata must have a static batch dimension equal to `B`;
- dynamic batch changes are unsupported and must not silently reuse state allocated for a
  different number of sequences.

`Config::pa_block_size` remains a paged-QSA configuration field and a logical kernel
parameter for the shared implementation; it is not the total dynamic dense cache length.
The lowering must not require `S == Config::pa_block_size` and must not reject a valid
single-batch dynamic sequence length solely because the current cache exceeds an initial
capacity hint. If the batch contract or dense layout cannot be proved, the graph pass must
fall back conservatively.

### 4.3 `QSAIndexerCache`

`QSAIndexerCache` is a GPU-internal stateful operation for the dense SDPA-style path. It
owns two independent, dynamically growing `Variable` states:

1. raw indexer keys;
2. compressed block summaries.

`VariableExtension` exposes only one variable per node, so raw-K is the variable returned by the generic interface and summary is exposed through explicit accessors:

- `get_variable()` / `get_variable_id()` — raw-K;
- `get_summary_variable()` / `get_summary_variable_id()` — summary.

This distinction is normative for state reset, cloning, lowering, and future beam-reorder logic.

#### Graph-level state declaration

`QSAIndexerCache` has no runtime data inputs and is not an independent Q1 producer. It
declares the identity of two persistent states and stores the graph-level `QSA::Config`:

1. raw-K state, represented by the generic `VariableExtension` accessors;
2. summary state, represented by explicit summary-variable accessors because
   `VariableExtension` exposes only one variable per node.

The operation therefore must not receive `MIXED`, indexer projection weights, RMSNorm
weights, `ROTARY_COS_SIN`, `PAST_LENS`, `SUBSEQUENCE_BEGINS`, or `POSITION_IDS`. Those
values are required by the actual Q1/Q2/Q3 computation in `QSASDPA`, not by the state
declaration. In particular, `QSAIndexerCache` does not recompute indexer Q/K, apply
RoPE, append raw K, or generate summaries as a separate graph operation.

The runtime `qsa_inst` owning the corresponding `QSASDPA` execution performs the
following state-management steps before and during the QSA dispatch:

- calculate the new raw-K and summary valid lengths from the current token count and
  `compress_ratio`;
- reserve or grow storage and relocate the valid old prefixes when necessary;
- update per-sequence base, stride, capacity, valid-length, phase, and reservation
  descriptors;
- execute the dense Q1/Q2/Q3 stages, with Q1 writing the raw-K and summary states;
- publish the new logical lengths only as part of the same runtime-state update.

There is no separate `cldnn::qsa_indexer_cache_state` primitive, `commit`/`finalize`
graph node, or `ReadValue -> prepare_only Q1 -> Assign` graph path in the target design.
The state declaration is analogous to `KVCache`; the runtime state manager is analogous
to the state handling inside `kv_cache_inst`.

The state contract still requires both Variables to share the same batch size `B` and
indexer width `D_i`. Runtime capacity may exceed valid length, and kernels must use the
published valid length rather than infer history from physical capacity. Reset, trim,
release, and future beam reorder must operate on raw-K and summary as one logical cache
contract.

#### Indexer-cache layout

The public state views are dense:

```text
raw-K state:   [B, L_raw,     Di]
summary state: [B, L_summary, Di]
```

Dense mode consumes the runtime descriptors directly. Per-sequence valid lengths and
position metadata identify the readable history, while physical capacity describes only
the current allocation. Raw K is retained in its pre-normalization/pre-RoPE form, while
summaries are pooled, K-normalized, and rotated at the block-start position. The two
Variables are one state contract: reset, clone, serialization, trim, release, and future
beam reorder must handle both together. The initial implementation is single-batch/
single-sequence decode; ragged batching, dynamic batch, and broader per-sequence capacity
management remain outside the current implementation contract.
## 5. Configuration and validation

`QSASDPA` reuses `QSA::Config` verbatim. Important fields are:

| Field | Meaning |
|---|---|
| `heads_num` | Main query head count, $H_q$ |
| `kv_heads_num` | Main KV head count, $H_{kv}$ |
| `k_head_size` | Main key width, $D_h$ |
| `v_head_size` | Main value width, $D_v$ |
| `hidden_size` | Indexer projection input width, $D$ |
| `indexer_heads` | $H_i$ |
| `indexer_kv_heads` | Indexer KV head count; current implementation requires MQA-compatible value |
| `indexer_head_dim` | $D_i$ |
| `indexer_rotary_dim` | Partial rotary width, no greater than $D_i$ |
| `budget` | Selected-token budget |
| `compress_ratio` | $R$; zero means the layer is not a QSA layer |
| `qsa_block_tokens` | Must be consistent with `compress_ratio` |
| `pa_block_size` | Paged-QSA page-size parameter; it is not the total length or capacity of a dynamic dense `QSASDPA` cache |
| `block_topk` | Must be consistent with budget and compression ratio |
| `scale` | Main attention scale, normally $1/\sqrt{D_h}$ |
| `indexer_norm_eps` | Indexer RMSNorm epsilon |
| `has_output_gate` | Whether query carries the output-gate component |
| `is_causal` | Causal attention flag |

Validation must reject inconsistent geometry, zero or incompatible dimensions, invalid
compression/budget relationships, unsupported page sizes, dynamic batch, and layouts that
cannot be proven to satisfy the dense cache contract. Dynamic sequence lengths and
runtime-growing raw-K/summary state are supported; they must not be rejected merely because
the current length differs from the initial allocation. The method returns a reason where
possible so a transformation can fall back to the original graph.

## 6. Internal kernel fusion and execution stages

The QSA primitive is divided into four ordered stages. The stages are a scheduling and
ownership boundary, not four independent OpenVINO operations: intermediate tensors may be
kept in cache or scratch memory, and the host runtime supplies the stage-specific maps and
offsets. The stages are:

| Stage | Kernel family | Main responsibility | Persistent state written |
|---|---|---|---|
| Q0 | `qsa_kv_cache_update` | Append the current projected K/V to the paged main cache | `KEY_CACHE`, `VALUE_CACHE` |
| Q1 | `qsa_prepare`, optionally `qsa_project_dpas` | Build indexer Q/K, apply normalization and RoPE, append raw K, and produce completed-block summaries | `INDEXER_RAW_K_CACHE`, `INDEXER_SUMMARY_CACHE` |
| Q2 | `qsa_score_*` | Score visible compressed blocks and select TopK candidates | Per-call score/TopK scratch |
| Q3 | `qsa_sparse_attention_*` | Read selected blocks and the causal tail, then evaluate sparse attention | Main output only |

The following subsections define the computation performed by each stage. The formulas use
one sequence slot; the implementation applies the same computation to ragged token ranges
described by `PAST_LENS` and `SUBSEQUENCE_BEGINS`.

### 6.1 Q0: main K/V cache update

Q0 receives the already projected current-token tensors `KEY` and `VALUE`. For sequence
slot $s$, token $t$, logical position $p = P_s + t$, and page size $P_A$, the logical
position is mapped to:

$$
\mathrm{page}(p) = \mathrm{BLOCK\_INDICES}[b_s + \lfloor p/P_A \rfloor],
\qquad
\mathrm{offset}(p) = p \bmod P_A,
$$

where $b_s$ is the offset from `BLOCK_INDICES_BEGINS`. For KV head $h$ and channel $d$:

$$
KCache[\mathrm{page}(p),h,\mathrm{offset}(p),d]
   \leftarrow K_{cur}[s,t,h,d],
$$

$$
VCache[\mathrm{page}(p),h,\mathrm{offset}(p),d]
   \leftarrow V_{cur}[s,t,h,d].
$$

Q0 is intentionally limited to cache update. It does not perform the Q/K/V projection,
attention score computation, or output projection. The page-table indirection is what makes
the ordinary `QSA` path suitable for externally scheduled continuous batching.

For `QSASDPA`, the upstream dense `KVCache` has already appended the current K/V. Therefore
Q0 is not scheduled (`skip_kv_write=true`). Dense mode uses the current resolved cache
layout and identity sequence metadata to reuse the common Q1/Q2/Q3 implementation; it does
not require a compile-time cache-length/page-size equality.

**Kernel partition:** `qsa_kv_cache_update.cm` owns only the writes above. It is scheduled
before Q1 and Q2/Q3 depend on its completion when `skip_kv_write` is false.

### 6.2 Q1: indexer preparation and summary maintenance

Q1 computes the indexer's current query and key representations and updates the compressed
history. Let the fused projection weight contain $H_iD_i$ query rows followed by $D_i$
key rows. For a mixed hidden state $x \in \mathbb{R}^{D}$:

$$
[q_{raw}, k_{raw}] = x W_{idx}^{T},
$$

where $q_{raw}$ has shape $H_i \times D_i$ and $k_{raw}$ has shape $1 \times D_i$.
The indexer applies RMS normalization independently to the Q and K legs:

$$
\widehat q_{h,j} = \frac{q_{raw,h,j}}{\sqrt{\mathrm{mean}_j(q_{raw,h,j}^{2})+\epsilon}}\,\gamma^q_j,
$$

$$
\widehat k_j = \frac{k_{raw,j}}{\sqrt{\mathrm{mean}_j(k_{raw,j}^{2})+\epsilon}}\,\gamma^k_j.
$$

Partial rotary embedding is then applied to the configured first `indexer_rotary_dim`
channels. For a rotary pair $(2j,2j+1)$ and position $p$:

$$
\mathrm{RoPE}(a,p)_{2j} = a_{2j}\cos\theta_{p,j} - a_{2j+1}\sin\theta_{p,j},
$$

$$
\mathrm{RoPE}(a,p)_{2j+1} = a_{2j}\sin\theta_{p,j} + a_{2j+1}\cos\theta_{p,j}.
$$

The remaining channels pass through unchanged. The resulting tensors are the indexer query
$q_i$ and the current raw key $k_i$.

Q1 appends $k_i$ at the logical history position. For each completed block $m$ of size
`compress_ratio` $R$, it forms the block summary:

$$
\bar{k}_{m} = \frac{1}{R}\sum_{u=0}^{R-1} k_i[mR+u].
$$

The summary is then normalized and, where configured, rotated using the block-start
position. Incomplete blocks are retained in the raw-K state but are not emitted as complete
summary candidates. This permits Q2/Q3 to apply the exact causal-tail rule instead of
rounding a partially visible block into a complete block.

Q1 is incremental: it starts from the cache position specified by `PAST_LENS`, processes
only the current token range, and updates only newly completed summaries. It does not
recompute the entire history on every call.

**Kernel partition:**

- `qsa_prepare.cm` implements projection, RMSNorm, rotary handling, raw-K writes, and the
  scalar/vector preparation path.
- `qsa_project_dpas.cm` is the DPAS projection variant used when device and geometry
  validation permit systolic execution.
- Both variants write the same logical raw-K and summary state and must remain numerically
  interchangeable within the documented tolerance.

`QSA` schedules Q1 as part of the complete operation. `QSASDPA` schedules the dense Q1,
Q2, and Q3 stages through the owning `qsa_inst`; Q1 writes the persistent raw-K/summary
state and also produces the current query-side indexer data needed by Q2. `QSAIndexerCache`
does not schedule a separate `prepare_only` dispatch. It declares the state identity and
provides the runtime descriptor used by the same `qsa_inst`.

### 6.3 Q2: block scoring and TopK selection

Q2 scores complete compressed blocks visible at the current causal position. For query head
$h$ and block summary $\bar{k}_m$:

$$
z_{s,h,m} = q_{i,s,h}^{T}\bar{k}_{s,m}.
$$

The QSA indexer applies ReLU before head reduction, then scales by the inverse square root
of the indexer head dimension:

$$
r_{s,h,m} = \mathrm{ReLU}(z_{s,h,m}),
$$

$$
score_{s,m} = \frac{1}{\sqrt{D_i}}
\sum_{h=0}^{H_i-1} r_{s,h,m}.
$$

Only blocks whose complete portion is causally visible are eligible for the complete-block
TopK pool. Depending on the export form, the causal restriction can be represented by a
mask/select before TopK or by the equivalent runtime score map. The implementation must
preserve the order of operations: ReLU precedes head reduction and the $1/\sqrt{D_i}$
scale follows the reduction.

Q2 selects `block_topk` indices, normally related to the token budget by:

$$
block\_topk = \left\lfloor\frac{budget}{R}\right\rfloor,
$$

subject to the operation's configuration validation. The selected physical blocks are
passed to Q3 through per-call score/index scratch and runtime dispatch maps. Q2 does not
modify the persistent main KV cache.

**Kernel partition:** `qsa_score_topk_fused.cm` is the fused score-and-TopK path for
supported shapes. `qsa_score_tile_dpas.cm` provides a tiled DPAS score path, while
`qsa_score_partition.cm` handles partitioned score work when the score matrix or scratch
budget requires chunking. The separate `qsa_topk_radix_wg.cm` kernel performs the radix
work-group TopK step when that implementation is selected. The host runtime selects the
variant and constructs the corresponding `score_tile_map`/partition metadata; the
mathematical result is the same score and TopK set.

### 6.4 Q3: selected-block sparse attention

Q3 evaluates the main attention only over the selected complete blocks and the causally
visible tail. Let $\mathcal{M}_s$ be the blocks selected by Q2, and let $\mathcal{T}_s$
be the visible positions in the current incomplete block. The effective key/value set is:

$$
\mathcal{V}_s = \left(\bigcup_{m\in\mathcal{M}_s} block_m\right)\cup\mathcal{T}_s.
$$

For query head $a$, mapped KV head $g(a)$, query vector $q_a$, key $k_{g(a),p}$, and
value $v_{g(a),p}$, Q3 computes:

$$
\ell_{a,p} = scale\; q_a^{T}k_{g(a),p} + mask_{a,p},
\qquad scale = \frac{1}{\sqrt{D_h}}.
$$

It then performs numerically stable online softmax over $p\in\mathcal{V}_s$:

$$
m_a = \max_{p\in\mathcal{V}_s}\ell_{a,p},
$$

$$
\alpha_{a,p} = \frac{\exp(\ell_{a,p}-m_a)}
{\sum_{u\in\mathcal{V}_s}\exp(\ell_{a,u}-m_a)},
$$

$$
y_a = \sum_{p\in\mathcal{V}_s}\alpha_{a,p}v_{g(a),p}.
$$

The causal tail is essential for prefill and decode correctness: a currently incomplete
block may not be in the summary TopK pool, but its visible token rows must still contribute.
GQA is handled by mapping multiple query heads to the corresponding KV head. If
`has_output_gate` is enabled, Q3 also applies the per-head gate carried by `QUERY` to the
attention result before producing the operation output.

**Kernel partition:** `qsa_sparse_attention_dpas.cm` is the DPAS implementation for the
selected-block path. `qsa_sparse_attention_gqa_dpas.cm` is the GQA-specialized DPAS variant.
`qsa_sparse_attention_split.cm` and `qsa_sparse_attention_split_reduce.cm` handle split
dispatch and reduction of partial online-softmax results when selected blocks or query rows
exceed one dispatch tile. The scalar/vector fallback is `qsa_sparse_attention.cm`. The
host runtime builds `token_map`, selected-block maps, and tail metadata; all variants
implement the same online-softmax and causal-tail semantics.

### 6.5 Stage ordering and synchronization

The normal paged execution order is:

```text
Q0: write current K/V cache
  -> Q1: compute indexer Q/K and update raw-K/summary state
  -> Q2: score visible summaries and select block_topk
  -> Q3: attend to selected blocks plus causal tail
```

Q1 may overlap independent preparation work only when its cache writes cannot race with Q2.
Q2 must observe the completed Q1 summary state, and Q3 must observe Q2's selected-block
maps. The implementation therefore chains stage events and uses internal scratch buffers
whose sizes are derived from the maximum dispatch maps.

For `QSASDPA`, the order is:

```text
upstream KVCache update
  -> qsa_inst IndexerCache.reserve()
  -> dense Q1: write raw-K/summary and prepare query-side indexer data
  -> Q2
  -> Q3
```

Q0 is omitted because the upstream dense KV cache owns the main K/V update. For
`QSAIndexerCache`, reserve/relocation and logical-length publication are runtime state
operations associated with the same `qsa_inst`; no separate graph execution is scheduled.

The design intentionally fuses at the operation/primitive boundary while keeping
surrounding projections as ordinary graph operations. This limits semantic ownership and
allows standard OpenVINO transformations to continue operating on projections, residual
paths, and output projections.

## 7. Graph transformations

### 7.1 `QSAFusion`

`QSAFusion` recognizes the indexer score signature:

```text
MatMul(q_index, k_block, transpose_b=true)
  -> Relu
  -> ReduceSum(axis=head)
  -> scale by 1/sqrt(Di)
  -> optional causal mask/clamp
  -> TopK
```

The pass additionally proves the required cache, projection, rotary, and page-table inputs and constructs `ov::intel_gpu::op::QSA`.

This is not an indexer-only replacement. The indexer score signature is the structural
anchor used to locate and prove the sparse-attention subgraph, while a successful rewrite
also proves the complete Q0-Q3 contract: current K/V inputs, page tables, main KV cache,
raw-K and summary caches, prepared main `QUERY`, and the downstream attention result.
Q0 and Q3 remain inside the resulting QSA primitive; they are not emitted as separate
OpenVINO nodes. Kernel stage boundaries do not imply graph-pass boundaries.

### 7.2 `QSASDPAFusion`

`QSASDPAFusion` recognizes the same indexer score signature and requires:

- Q and K indexer legs derived from one shared projection;
- an indexer raw-K history backed by `ReadValue`/`Variable`;
- a downstream `ScaledDotProductAttention` node;
- main K and V operands backed by distinct cache state;
- statically provable batch and dense cache ownership/layout, while allowing runtime sequence growth;
- one already-packed rotary table in the supported representation.

On success it creates the state declaration and the attention operation:

```text
QSAIndexerCache (state identity/configuration)
                 \
                  -> QSASDPA/qsa_inst -> attention output
```

`QSASDPAFusion` is a complete replacement. The graph contains a state declaration and an
attention operation, while runtime ownership is unified in the corresponding `qsa_inst`:

- `QSAIndexerCache` declares persistent raw-K/summary state identity and configuration;
- `qsa_inst` owns reserve, relocation, valid lengths, descriptors, and state publication;
- `QSASDPA`/`qsa_inst` executes Q1, Q2 scoring, and Q3 sparse attention.

Q0 is intentionally absent from `QSASDPA`: the upstream dense KV cache already owns the
current K/V append. Thus the pass is not incomplete merely because Q0 is not constructed
as a separate node. It is a dense-cache lowering in which Q0 is represented by cache
ownership, while Q1/Q2/Q3 are preserved through the paired operation contract. The pass
does not fuse arbitrary `q_proj`, `k_proj`, `v_proj`, or `o_proj` nodes; those remain
ordinary graph operations.

For the current real Qwen export, separate `full_cos`/`full_sin` parameters are an
intentional bail-out condition because the supported contract requires one packed rotary
table. Dynamic sequence/cache dimensions are not, by themselves, a bail-out: the pass may
replace the graph when the batch is statically provable and the main `KVCache` plus both
indexer Variables have the required ownership and dense layout. Dynamic batch remains a
conservative rejection/fallback condition. Graph-side rotary repacking is future work.

## 8. Performance design

### 8.1 Kernel reuse

The primary optimization is reuse of the existing QSA CM implementation. This avoids a
second numerical implementation of sparse attention and preserves the validated DPAS,
online-softmax, and causal-tail code.

### 8.2 Avoided work

The runtime state manager avoids reallocating or copying the complete raw-K history and its
block summaries on every call. It reserves only the required new valid length; Q1 processes
the new token range and newly completed blocks, writes the persistent states, and produces
the current query-side indexer data needed by Q2. There is no separate Q1 execution in a
`QSAIndexerCache` graph node and therefore no duplicate Q1 work caused by that node.

### 8.3 Degenerate-page cost

The dense lowering uses one logical contiguous sequence region per batch slot. Compared
with true paging, this can increase memory reservation and Q2 scratch upper bounds for
large batches. Runtime state and scratch buffers are reused when capacity is sufficient
and grown when the resolved layout exceeds the previous allocation. It is efficient for
single-stream and small fixed-batch deployments, while paged `QSA` remains preferred for
scheduler-driven continuous batching or dynamic batch management.

### 8.4 Dispatch reuse

`PAST_LENS` and `SUBSEQUENCE_BEGINS` are retained in `QSASDPA` to preserve the existing
`PREFILL`, `GENERATE`, and `MIXED` host dispatch paths. Dispatch scalars and scratch/state
descriptors are refreshed from the resolved runtime layouts on shape changes; reuse is
bounded by the currently allocated capacity rather than by a fixed compile-time sequence
length.

## 9. OpenVINO conformance and integration requirements

The implementation follows these OpenVINO conventions:

- public/internal operation classes are separate from Intel GPU `gpu_opset` wrappers;
- `OPENVINO_OP` and `OPENVINO_MATCHER_PASS_RTTI` identify operations and passes;
- `validate_and_infer_types()` supplies output type and shape inference;
- `clone_with_new_inputs()` preserves the concrete GPU operation type;
- `visit_attributes()` serializes operation attributes;
- `Config` validation is explicit and reusable;
- graph passes are conservative and preserve generic fallback graphs;
- stateful nodes expose variable identity and support clone/state-management requirements;
- ProgramBuilder factory registration is separate for each operation;
- no undocumented runtime state is hidden in a stateless attention operation;
- ordinary model inputs remain ordinary OpenVINO graph inputs unless a transformation proves ownership transfer.

The operation should remain plugin-internal until API stability, serialization, shape-inference, CPU/reference behavior requirements, and cross-plugin ownership are approved by the OpenVINO maintainers. The current implementation is therefore marked under development in the core declarations.

## 10. Future work

The current single `QSA` primitive is a deliberate version-1 implementation boundary, not
the only possible decomposition. For extensibility, QSA can be split into two primitives:

1. **Indexer primitive:** owns Q1 projection/normalization/RoPE, raw-K append, summary
  maintenance, Q2 block scoring, and TopK selection. Its outputs are selected block IDs,
  counts/maps, and optionally persistent raw-K/summary state.
2. **Sparse-attention primitive:** owns Q0 main KV-cache update and Q3 selected-block plus
  causal-tail attention. It consumes prepared main `QUERY`, current K/V, page tables,
  selected-block metadata, and the main KV cache.

This split permits Q1/Q2 to be cached, reused, or run at a different cadence from Q0/Q3.
It also requires explicit versioned cache state, selection metadata, and synchronization
between Q0 writes, indexer updates, and Q3 reads. The split must preserve the current
semantics: Q2 selects complete compressed blocks, Q3 adds the causal tail, and Q0 writes
the physical page layout.

Other planned extensions are:

1. **Packed rotary-table normalization:** support graph-side conversion from separate cosine
  and sine parameters after numerical and shape proofs are available.
2. **Dynamic batch:** support for changing the number of dense sequence slots remains
  future work; until then, reject it or fall back to paged `QSA`. Dynamic sequence lengths,
  cache growth, and indexer-state growth are already supported for a fixed batch.
3. **Unequal per-sequence capacities:** support shared-capacity padding with explicit
  validation and tests.
4. **Beam reorder:** coordinate main K/V, raw-K, and summary state. The safe first
  implementation is eager `Gather`; a later kernel-level indirect mode can avoid copies.
5. **Q1 de-duplication:** expose query-side indexer computation without duplicating persistent
  cache preparation.
6. **Memory-budget controls:** add capacity hints, trim/sliding-window behavior, or select
  paged QSA when dense reservation is not acceptable.
7. **Reference and serialization coverage:** add broader model-import, clone, save/read, and
  fallback tests before promoting the operation beyond plugin-internal status.

## 11. Verification plan and current evidence

The test plan must include:

- configuration and input-count validation;
- output shape/type inference;
- clone and attribute round trips;
- paged QSA regression tests;
- dense QSASDPA prefill and decode numerical comparisons;
- mixed dispatch tests;
- dynamic sequence acceptance and cache-growth tests across repeated inference calls;
- explicit dynamic-batch rejection/fallback tests;
- QSAIndexerCache persistence, relocation/copy-forward, and summary-boundary tests across
  two or more independent inference calls;
- reset-state behavior for both cache variables;
- fusion success and conservative bail-out cases;
- real-model topology checks;
- performance measurements for memory, Q1 duplication, scratch size, and latency.

The current repository contains host-side operation tests, real Intel GPU lowering tests for
QSASDPA, repeated-inference indexer-cache growth tests, and graph-transformation tests.
These tests establish fixed-batch dynamic sequence support; dynamic batch, unequal
per-sequence capacity/padding, beam reorder, and rotary-table repacking remain explicit
coverage gaps rather than assumed support.

## 12. oneDNN applicability and requested support

QSA is an Intel GPU operation whose current implementation relies on CM kernels for page
table indirection, compressed-block selection, and GPU-specific DPAS paths. oneDNN can
cover several surrounding computations or provide reusable building blocks, but replacing
the whole QSA primitive with existing dense oneDNN operations would not preserve the
current sparse-cache contract.

### 12.1 Suitable for oneDNN or standard primitives

| Computation | Possible mapping | Boundary/limitation |
|---|---|---|
| Dense indexer projection `MIXED * W_idx^T` | GEMM/MatMul | Does not include page addressing or state update. |
| Indexer Q/K RMSNorm | RMSNorm or primitive composition | Epsilon, gain, and accumulation semantics must match. |
| Block mean pooling | Reduction/mean | Must avoid forcing a full-history recomputation in the incremental path. |
| Indexer dot products and head reduction | MatMul plus ReduceSum | ReLU-before-reduction and scale ordering must be preserved. |
| Sparse Attention | Custom kernel or new oneDNN primitive | Requires paged-sparse handling, causal tail addition, and online softmax without materializing a dense matrix. |
|

These mappings are useful for CPU/reference paths, validation kernels, or
future hybrid implementations. They should not be assumed to match CM memory traffic or
numerical ordering automatically.

### 12.2 Functionality requiring new oneDNN support

The following capabilities require a paged-sparse extension rather than another dense
GEMM/SDPA option:

1. **Paged KV-cache update:** write current K/V using per-sequence physical page tables.
2. **Paged sparse attention:** gather selected compressed blocks, add the causal tail, and
  perform numerically stable online softmax without materializing a dense matrix.
3. **Fused indexer state update:** append raw K and update only newly completed summaries in
  persistent state with ragged `PAST_LENS` and `SUBSEQUENCE_BEGINS`.
4. **Fused score-and-TopK:** combine summary loads, ReLU, head reduction, causal eligibility,
  and TopK while producing metadata consumable by Q3.
5. **QSA-specific persistent state:** coordinate raw-K and summary Variables, reset, clone,
  beam reorder, and cross-infer-request lifetime.

### 12.3 Q3-specific gap if implemented with oneDNN

Q3 is the largest gap for a direct oneDNN replacement. Its computation is not a standard
dense SDPA over one contiguous sequence. For each query token it consumes a different
`selected_blocks` list, translates each logical block through `BLOCK_INDICES` into a
physical page, reads only the selected complete blocks plus the visible causal tail, and
performs one numerically stable online-softmax reduction. The current CM kernel does this
without materializing selected K/V or a dense attention mask.

| Q3 requirement | Why an existing dense oneDNN SDPA is insufficient | Support required for a native oneDNN path |
|---|---|---|
| Per-token `selected_blocks` and `selected_counts` | A normal SDPA call assumes one dense K/V domain and cannot express a different sparse block set for every query token. | A sparse attention descriptor with per-query block metadata and invalid-slot handling. |
| Two-level page indirection through `BLOCK_INDICES` and `BLOCK_INDICES_BEGINS` | Dense strides cannot represent logical-to-physical page translation during the K/V read. | Device-side page-table lookup, page size, and bounds-safe indirect K/V loads. |
| Selected complete blocks plus an incomplete causal tail | A conventional causal mask describes a triangular region, not “TopK complete blocks plus positions through `abs_pos`”. | Separate selected-block and tail regions, with exact causal eligibility per sequence/token. |
| GQA mapping `kv_head = query_head / (Hq/Hkv)` | Generic attention may support grouped heads, but the sparse page read must preserve the Q/KV head mapping for every indirect load. | Explicit Hq/Hkv mapping in the primitive descriptor and kernel dispatch. |
| Streaming online softmax | Materializing scores or using a dense softmax increases memory traffic and loses the current sparse execution model. | Fused QK, max/sum update, V accumulation, and final normalization with documented accumulation type. |
| QSA block-contiguous fetches | A generic SDPA tile layout may fetch individual tokens or dense tiles, wasting bandwidth at high sparsity. | A block gather path whose layout is aligned with `PA_BLOCK_SIZE` and compression ratio `r`. |
| Optional per-channel output gate | The gate is carried beside `QUERY` and is applied after attention accumulation, before the output leaves Q3. | An optional fused gate input/post-op, or an explicit post-attention graph operation with matching layout. |
| Ragged tokens and runtime dispatch metadata | `PAST_LENS`, `SUBSEQUENCE_BEGINS`, and page-table-begin offsets vary by sequence and request. | Ragged batch metadata as first-class runtime inputs, not compile-time constants. |

There are two possible implementation levels:

1. **Compatibility path:** gather the selected K/V blocks into temporary dense tensors,
   construct a per-token mask, and call a dense oneDNN SDPA. This can provide a reference
   or correctness fallback, but it loses the main Q3 benefits: no materialized selected
   K/V, no page-aware coalescing, higher temporary-memory use, and extra gather/mask
   bandwidth. It is therefore not performance-equivalent to the current CM Q3 kernel.
2. **Native path:** add a oneDNN paged-sparse attention primitive that accepts page tables,
   selected-block metadata, causal-tail bounds, GQA dimensions, and optional gate data.
   The primitive must guarantee f16 cache/query storage with the required f32 accumulation,
   stable online softmax, deterministic invalid-slot behavior, and a capability query for
   supported page size, head sizes, compression ratio, and data types.

Consequently, “run Q3 on oneDNN” is feasible as a dense compatibility fallback today only
if the graph accepts the materialization cost. It is not a drop-in replacement for the
current paged CM implementation. To preserve Q3's intended performance model, oneDNN
would need the native sparse primitive described above; until then, Q3 should remain in
the Intel GPU CM path while oneDNN handles dense fallback or surrounding operations.

A useful oneDNN request would therefore be an explicit **paged sparse attention and cache
state API**, not a dense SDPA flag. It would need page size and page-table inputs, separate
append/read phases, selected-block metadata, ragged causal semantics, optional auxiliary
state, documented f16-storage/f32-accumulation behavior, and a capability query. Until
such an API exists, Q0-Q3 should remain in the Intel GPU CM implementation while standard
oneDNN/OpenVINO operations handle surrounding projections, normalization, reductions, and
dense fallback paths.
