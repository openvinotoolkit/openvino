.. {#openvino_docs_ops_internal_QSAIndexer}

QSAIndexer
==========

.. meta::
  :description: Learn about QSAIndexer - the standard, stateful block-level indexer for Qwen Sparse Attention (QSA). It consumes projected raw query and key tokens, per-token RoPE cos/sin, plus writable full-history raw-key, summary and block-start RoPE caches, updates them in place, and emits selections containing only complete blocks.

**Versioned name**: *QSAIndexer*

**Category**: *Internal*

**Short description**:
The *QSAIndexer* is the **model-specific plan producer** for Qwen Sparse Attention (QSA) in its
**standard, non-paged, stateful** form. It consumes externally projected indexer queries and keys, applies query
RMSNorm/RoPE (with caller-supplied per-token ``rope_cos``/``rope_sin``), appends raw keys, generates summaries for
completed blocks, and produces block selections. The
caller supplies writable raw-key, summary and RoPE cache tensors; the operation updates these buffers in place and does
not return cache tensors as outputs. This makes it suitable for **single-sequence / fixed-batch**
SDPA pipelines that do not use a paged KV cache. For continuous batching over a PagedAttention page table, see the separate
:doc:`PagedQSAIndexer <paged-qsa-indexer>` operation.

**State and in-place update contract**

Unlike the paged variant, *QSAIndexer* reads **no page table** and owns no opaque runtime state. Its logical cache
state consists of three caller-owned writable tensors: ``summary_cache`` stores completed-block summaries,
``indexer_raw_k_cache`` stores the full raw-key history, and ``indexer_rope_cache`` stores the RoPE coefficients of
each block's first token. The cache dimensions describe allocated capacity; the
``past_lens`` input describes how much of each raw-key cache is valid before the invocation. The caller must
provide non-aliasing writable buffers and serialize invocations that update the same state. The operation mutates
the buffers but does not resize them or return replacement caches.

For beam branching, callers must provide independent writable cache storage for each branch before invoking the
operation. For rollback, the caller restores ``past_lens``; cache entries beyond the restored valid length are
ignored and overwritten when replayed. The operation is stateful and must not be eliminated, duplicated, or
reordered across updates to the same cache state.

**Detailed description**

The *QSAIndexer* consumes externally projected indexer queries and keys; projection weights are not inputs. It
applies query RMSNorm and partial RoPE, appends unnormalized and unrotated keys to the raw-key state, generates
summaries for completed blocks, and performs block scoring and top-k selection. The summary chain is
*non-associative* and *incremental*:
:math:`\mathrm{RMSNorm}(\mathrm{Mean}(K)) \ne \mathrm{Mean}(\mathrm{RMSNorm}(K))`, and a block summary can only be
maintained incrementally as new tokens arrive.

**Head layout and the index-head grouping**

The indexer distinguishes two head dimensions that are **derived from the input shapes**, not from an attribute:

- ``q_idx`` carries ``H_iq`` index-*query* heads in its penultimate dimension (``[B, L, H_iq, D_idx]``).
- ``k_idx`` carries ``H_ik`` index-*key* heads in its penultimate dimension (``[B, L, H_ik, D_idx]``).

``H_ik`` is configurable and taken from the ``k_idx`` shape, never hard-coded to ``1``. This lets a single block
summary key be stored per index-key head, so models that need more than one key projection head are expressible
without adding a special-case attribute. The score for selection head ``sh`` is a grouped, per-query-head ReLU
reduction. The grouping is fully determined by the head counts:

.. math::

    H_{iq} \bmod H_{ik} = 0, \quad G = H_{iq} / H_{ik}, \quad \mathrm{kv}(h) = \lfloor h / G \rfloor, \quad
    \mathrm{sel}(h) = \lfloor \mathrm{kv}(h) \cdot H_{sel} / H_{ik} \rfloor

    s_{b, sh} = \mathrm{scale} \cdot \sum_{\substack{h \in [0, H_{iq}) \\ \mathrm{sel}(h) = sh}}
        \mathrm{ReLU}\!\left( \langle q_{h}, \bar{k}_{b, \mathrm{kv}(h)} \rangle \right)

where:

- ``H_iq % H_ik == 0`` and :math:`G = H_{iq} / H_{ik}` is the number of index-*query* heads sharing one
  index-*key* head,
- :math:`\mathrm{kv}(h) = \lfloor h / G \rfloor` maps index-query head :math:`h` to its index-key head
  :math:`\mathrm{kv}(h) \in [0, H_{ik})`,
- :math:`H_{sel} \in \{1, H_{ik}\}` is the number of selection heads, and
- :math:`\mathrm{sel}(h) = \lfloor \mathrm{kv}(h) \cdot H_{sel} / H_{ik} \rfloor` maps the index-key head of
  query head :math:`h` to its selection head.

:math:`\bar{k}_{b, \mathrm{kv}(h)}` is the per-index-key-head summary key of block :math:`b`. The ReLU is applied
per index-query head before reduction over heads, as specified by the QSA scoring semantics. Under QSA
(:math:`H_{iq} = 4`, :math:`H_{ik} = 1`, :math:`H_{sel} = 1`) all four index-query heads map to the single
index-key head and the single selection head, so the score is the sum over all four per-head ReLU dot-products.

**State organization**

The logical cache state is stored in three preallocated, in-place updated tensors. Let :math:`S_{cap}` be the raw-key
cache capacity:

- ``summary_cache`` ``[B, H_ik, floor(S_cap / r), D_idx]`` stores mean-pooled, RMSNorm'ed, first-token-RoPE'ed
  summaries for complete blocks. For batch element ``b``, entries ``[0, floor(past_lens[b] / r))`` are valid
  before the invocation. Newly completed summaries are written into this same tensor; old valid entries are not
  recomputed during ordinary incremental execution.
- ``indexer_raw_k_cache`` ``[B, H_ik, S_cap, D_idx]`` stores all projected, unnormalized, unrotated per-token
  indexer keys. For batch element ``b``, entries ``[0, past_lens[b])`` are valid before the invocation. Current
  keys are written at their logical positions ``past_lens[b] + l``. This full-history layout uses :math:`O(S_{cap})`
  storage and permits direct reread/replay of historical raw keys.
- ``indexer_rope_cache`` ``[B, ceil(S_cap / r), indexer_rotary_dim]`` stores, for each block, the RoPE
  coefficients of its first logical token as one interleaved row ``(cos_0, sin_0, cos_1, sin_1, ...)``. Its
  capacity is ``ceil`` rather than ``floor``: when ``S_cap % r != 0``, the trailing partial block still starts
  inside the raw-key capacity and needs a row. For batch
  element ``b``, rows ``[0, ceil(past_lens[b] / r))`` are valid before the invocation (this includes a pending
  block whose first token was appended by an earlier invocation). A row is written once, when its block's first
  token is appended, and read back once when the block completes.

The caller ensures capacity before invocation and advances ``past_lens`` by ``L`` after a successful call. The
precondition is ``past_lens[b] + L <= S_cap`` for every batch element; the operation does not resize any cache.
Every one of the ``B x L`` query tokens is a real token: there is no padding and no per-row valid length, so each
batch row appends exactly ``L`` new tokens per invocation (``past_lens`` may still differ between rows). Rows with
different new-token counts must be issued as separate invocations or use
:doc:`PagedQSAIndexer <paged-qsa-indexer>`.
The **logical position** of query token ``l`` in batch element ``b`` is ``past_lens[b] + l``; it drives block
addressing and causal filtering only, never a RoPE angle. Query token ``(b, l)`` is rotated with
``(rope_cos[b, l], rope_sin[b, l])``; the summary of block ``j`` is rotated with the coefficients of the token at
logical position ``j * r`` (its first token), read from ``indexer_rope_cache`` rather than recomputed from
``j * r``. Because the per-token cos/sin are supplied by the producer (normally the same tensors the main attention
RoPE consumes), any position scheme -- mRoPE sections, non-monotonic or gapped positions, scaled RoPE variants --
is reproduced exactly without an op change. A block is
*completed* once exactly :math:`r` raw keys have been written. The last partially-filled block is not
summarized and is never included in this operation's selection. The paired sparse-attention consumer derives the
causal tail directly from each query's logical position and processes its visible token prefix separately.

**Score computation**

For a query at logical position :math:`p` with projected query :math:`q_h \in \mathbb{R}^{D_{idx}}` (per index-query
head :math:`h`) and block :math:`b` whose per-index-key-head summary key is
:math:`\bar{k}_{b, \mathrm{kv}(h)} \in \mathbb{R}^{D_{idx}}`, the per-selection-head score is the grouped reduction
over index-query heads (see the formula above), mapping each query head to its selection head via
:math:`\mathrm{sel}(h)`. The ReLU is applied *per index-query head before reduction over heads*, matching the
QSA scoring semantics. The indexer query :math:`q` arrives projected but not normalized
or rotated. The op applies query RMSNorm (via ``q_norm_weight``) and partial RoPE using the token's own
``(rope_cos, rope_sin)`` row. The pooled block key undergoes RMSNorm (via ``k_norm_weight``) and partial RoPE
using its first token's row from ``indexer_rope_cache``. The rotation is NEOX rotate-half over the first
:math:`R = \mathrm{indexer\_rotary\_dim}` channels, computed in ``fp32``:

.. math::

    x'_j = x_j \cos_j - x_{j + R/2} \sin_j, \quad x'_{j + R/2} = x_{j + R/2} \cos_j + x_j \sin_j,
    \quad 0 \le j < R/2

The RMSNorm used for the pooled block key is defined as

.. math::

    \mathrm{rms\_norm}(x, w, \mathrm{eps}) = x \cdot \mathrm{rsqrt}\!\left( \mathrm{mean}(x^2) + \mathrm{eps} \right) \cdot w

computed in ``fp32`` regardless of the input precision. Qwen's Gemma-style scale :math:`(1 + w)` is folded by the
model converter into the stored weight :math:`w' = 1 + w`, so the op consumes the already-shifted weight.

**Numerics of the block summary.** The pooled key is accumulated in ``fp32`` and rounded to *T* before RMSNorm,
matching the reference model (``mean(k.float()).to(k.dtype)``); RMSNorm and RoPE are then evaluated in ``fp32``
and the result is stored to ``summary_cache`` as *T*. An implementation MAY keep the pooled mean in ``fp32``
instead of rounding it to *T*; the summaries then differ by at most one rounding of *T*, which can change the
selection only among blocks whose scores are equal within that error.

**Score scale.** ``scale`` multiplies every score of a query by the same positive constant, so it never changes
the ranking: in exact arithmetic ``sel_indices`` and ``sel_count`` do not depend on its value. It is kept for
parity with the reference scoring (``1 / sqrt(indexer_head_dim)``); an implementation MAY omit the
multiplication.

**Selection, tie-breaking, and top-k**

- A block is *causally valid* for a query at logical position :math:`p` iff its last token is at most :math:`p`,
  i.e. :math:`N_{complete}(p) = \lfloor (p + 1) / r \rfloor` blocks are eligible. Only these causally-valid
  complete blocks participate in the top-k (future / not-yet-written blocks never leak into the selection).
- The number of selected *complete* blocks per selection head is capped at ``block_topk``; the output width is
  ``K_max = block_topk``. The selection contains no causal-tail block ID, and ``sel_count`` counts complete
  selected blocks only: ``0 <= sel_count <= min(block_topk, n_complete)``.
- **Tie-breaking and ordering**: among equal scores the *smaller block index* (earlier block) wins; the op
  computes the selection as ``argsort(-scores, kind='stable')``, takes the first ``k`` entries, and emits them in
  **strictly ascending** block order (``sort(order[:k])``), yielding a deterministic, implementation-independent,
  monotone result. The valid output prefix ``sel_indices[t, sh, :sel_count[t, sh]]`` is sorted ascending.

**Block grid and causal rules**

The KV history of a batch element is partitioned into blocks of :math:`r` tokens. Block :math:`b` covers tokens
:math:`[b \cdot r, (b+1) \cdot r - 1]`. Scoring is causal: a query at logical position :math:`p` may only select
complete blocks with :math:`b < \lfloor (p+1)/r \rfloor`. The output contains only these selected complete-block
IDs; neither a tail ID nor tail token indices are emitted. For causal attention, the consumer independently
computes :math:`tail\_start = \lfloor (p+1)/r \rfloor r` and attends token positions
:math:`tail\_start, \ldots, p` (inclusive). This tail is empty when :math:`(p+1) \bmod r = 0`. The consumer
applies this rule per query and also enforces the token-level causal bound :math:`kpos \le q_{abs}`.

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def apply_partial_rope(x, cos, sin, rotary_dim):
      # NEOX rotate-half over the first rotary_dim channels, in fp32; cos/sin: [..., rotary_dim // 2]
      half = rotary_dim // 2
      x = x.astype(float32)
      x1, x2 = x[..., :half].copy(), x[..., half:rotary_dim].copy()
      x[..., :half] = x1 * cos - x2 * sin
      x[..., half:rotary_dim] = x2 * cos + x1 * sin
      return x

    def qsa_indexer(q_raw, k_raw, rope_cos, rope_sin, summary_cache,
            indexer_raw_k_cache, q_norm_weight, k_norm_weight, past_lens,
            indexer_rope_cache,
            *, compress_ratio, block_topk, h_sel, indexer_rotary_dim,
            indexer_head_dim, eps, scale):
      # q_raw: [B, L, H_iq, D_idx], external projection output, not normed/RoPE'd
      # k_raw: [B, L, H_ik, D_idx], external projection output, unnormed/unrotated
      # rope_cos, rope_sin: [B, L, indexer_rotary_dim // 2], per-token NEOX coefficients
      # summary_cache: [B, H_ik, floor(S_cap/r), D_idx], writable in place
      # indexer_raw_k_cache: [B, H_ik, S_cap, D_idx], writable in place
      # past_lens: [B], valid raw-key lengths before this invocation
      # indexer_rope_cache: [B, ceil(S_cap/r), indexer_rotary_dim], interleaved (cos_j, sin_j), writable
      B, L, H_iq, Di = q_raw.shape
      Hik = k_raw.shape[2]
      Hsel, r = h_sel, compress_ratio
      Kmax = block_topk
      G = H_iq // Hik
      S_cap = indexer_raw_k_cache.shape[2]
      assert H_iq % Hik == 0 and Hsel in (1, Hik) and r >= 1
      assert summary_cache.shape[2] == S_cap // r
      assert indexer_rope_cache.shape[1] == (S_cap + r - 1) // r
      assert past_lens.shape == (B,)

      q = rms_norm(q_raw, q_norm_weight, eps)
      q = apply_partial_rope(q, rope_cos[:, :, None, :], rope_sin[:, :, None, :], indexer_rotary_dim)

      sel_indices = full((B, Hsel, L, Kmax), -1, dtype=int32)
      sel_count = zeros((B, Hsel, L), dtype=int32)
      for batch in range(B):
        S0 = past_lens[batch]
        assert 0 <= S0 and S0 + L <= S_cap
        assert summary_cache.shape[2] >= (S0 + L) // r
        for l in range(L):
          indexer_raw_k_cache[batch, :, S0 + l, :] = k_raw[batch, l, :, :]
          pos = S0 + l
          if pos % r == 0:  # first token of block pos // r: record its RoPE coefficients once
            indexer_rope_cache[batch, pos // r, 0::2] = rope_cos[batch, l]
            indexer_rope_cache[batch, pos // r, 1::2] = rope_sin[batch, l]

        n_complete_present = (S0 + L) // r
        for b in range(S0 // r, n_complete_present):
          grp = indexer_raw_k_cache[batch, :, b*r:(b+1)*r, :]
          pooled = mean(grp.astype(float32), axis=1).astype(k_raw.dtype)
          normed = rms_norm(pooled, k_norm_weight, eps)
          row = indexer_rope_cache[batch, b]  # coefficients of the block's first token
          summary_cache[batch, :, b, :] = apply_partial_rope(normed, row[0::2], row[1::2], indexer_rotary_dim)

        for l in range(L):
          pos = S0 + l
          n_complete = (pos + 1) // r
          for sh in range(Hsel):
            scores = score_complete_blocks(q[batch, l], summary_cache[batch],
                             n_complete, sh, G, scale)
            k = min(block_topk, n_complete)
            order = stable_argsort_descending(scores)  # equal scores: smaller block ID first
            picked = sort(order[:k])                   # output IDs are ascending
            sel_indices[batch, sh, l, :k] = picked
            sel_count[batch, sh, l] = k
      return sel_indices, sel_count


**Attributes**

* *compress_ratio*

  * **Description**: :math:`r`, the number of raw token keys pooled into one block key.
  * **Type**: ``int``
  * **Default value**: ``4``
  * **Required**: *no*
  * **Constraints**: ``compress_ratio >= 1``.

* *block_topk*

  * **Description**: Maximum number of *complete* blocks selected per selection head and the output width
    ``K_max``. The causal tail is not part of this output and does not consume a selection slot.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``block_topk >= 1``.

* *h_sel*

  * **Description**: Number of selection heads :math:`H_{sel}` in the emitted contract. Must be ``1`` or
    ``H_ik`` (the index-key head count taken from ``k_idx``). Each index-query head :math:`h` is mapped to a
    selection head via :math:`\mathrm{sel}(h) = \lfloor \mathrm{kv}(h) \cdot H_{sel} / H_{ik} \rfloor`, where
    :math:`\mathrm{kv}(h) = \lfloor h / G \rfloor` and :math:`G = H_{iq} / H_{ik}`.
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*
  * **Constraints**: ``h_sel >= 1``; ``h_sel == 1`` (QSA default) or ``h_sel == H_ik``.

* *indexer_rotary_dim*

  * **Description**: RoPE dimension used by the indexer (partial rotary). Only the first
    ``indexer_rotary_dim`` channels of each query head and block summary are rotated; the trailing
    ``indexer_head_dim - indexer_rotary_dim`` channels are passed through unrotated. ``rope_cos`` / ``rope_sin``
    carry ``indexer_rotary_dim / 2`` coefficients per token; an ``indexer_rope_cache`` row carries
    ``indexer_rotary_dim`` interleaved values.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: even, ``indexer_rotary_dim <= indexer_head_dim``.

* *indexer_head_dim*

  * **Description**: Head dimension :math:`D_{idx}` of the indexer.
  * **Type**: ``int``
  * **Required**: *yes*

* *eps*

  * **Description**: Epsilon for RMSNorm.
  * **Type**: ``float``
  * **Default value**: ``1e-6``
  * **Required**: *no*

* *scale*

  * **Description**: Positive multiplier applied to every block score (see "Score scale"). It does not affect
    the selection outputs.
  * **Type**: ``float``
  * **Default value**: ``1.0 / sqrt(indexer_head_dim)``
  * **Required**: *no*
  * **Constraints**: ``scale > 0``.


**Inputs**

* **0**: ``q_idx``
  A 4D tensor of type *T* with shape ``[B, L, H_iq, D_idx]``.
  Raw projected indexer queries before indexer RMSNorm and RoPE. The operation applies both internally. **Required.**

* **1**: ``k_idx``
  A 4D tensor of type *T* with shape ``[B, L, H_ik, D_idx]``.
  Current-step projected indexer keys. The index-key head count ``H_ik`` is derived from this shape and may be
  greater than one. Keys are appended to the
  full-history raw-K cache and pooled into newly completed-block summaries. **Required.**

* **2**: ``rope_cos``
  A 3D tensor of type *T* with shape ``[B, L, indexer_rotary_dim / 2]``. ``rope_cos[b, l, j]`` is the cosine of
  NEOX (rotate-half) frequency ``j`` for query token ``(b, l)``, rotating channels ``j`` and
  ``j + indexer_rotary_dim / 2``. Intended to be the same per-token cosine tensor consumed by the main attention
  RoPE; the operation does not recompute angles, so producers should evaluate them in ``fp32`` before any cast
  to *T*. **Required.**

* **3**: ``rope_sin``
  A 3D tensor of type *T* with shape ``[B, L, indexer_rotary_dim / 2]``; the sine counterpart of ``rope_cos``
  with the same layout. **Required.**

* **4**: ``summary_cache``
  A writable 4D tensor of type *T* with shape ``[B, H_ik, floor(S_cap / compress_ratio), D_idx]``.
  Caller-allocated storage for completed-block summaries. The operation reads valid summaries and writes only
  summaries for newly completed blocks, in place; it does not allocate, reserve, or grow this cache. **Required.**

* **5**: ``indexer_raw_k_cache``
  A writable 4D tensor of type *T* with shape ``[B, H_ik, S_cap, D_idx]``. It stores the full history of
  projected, unnormalized, unrotated indexer keys. The operation reads the valid history and writes only current-step
  keys at their logical positions, in place; it does not allocate, reserve, or grow this cache. **Required.**

* **6**: ``q_norm_weight``
  A 1D tensor of type *T* with shape ``[D_idx]``. RMSNorm weight for raw projected indexer queries. **Required.**

* **7**: ``k_norm_weight``
  A 1D tensor of type *T* with shape ``[D_idx]``.
  RMSNorm weight for pooled block keys. **Required.**

* **8**: ``past_lens``
  A 1D integer tensor with shape ``[B]``. For each batch element, gives the number of valid raw keys in
  ``indexer_raw_k_cache`` before this invocation. This input is read-only; the operation does not advance or otherwise
  modify it. The caller advances it by ``L`` only after successful execution. **Required.**

* **9**: ``indexer_rope_cache``
  A writable 3D tensor of type *T* with shape ``[B, ceil(S_cap / compress_ratio), indexer_rotary_dim]``. One row
  per QSA block holding the RoPE coefficients of that block's first logical token, interleaved as
  ``(cos_0, sin_0, cos_1, sin_1, ...)``. The operation writes ``row[2j] = rope_cos[b, l, j]`` and
  ``row[2j + 1] = rope_sin[b, l, j]`` into ``indexer_rope_cache[b, pos // r]`` once, when logical position
  ``pos = past_lens[b] + l`` is appended with ``pos % r == 0``, and reads it back once the block completes to rotate
  that block's summary, regardless of which invocation wrote it. **Required.**


**Outputs**

* **0**: ``sel_indices``
  A 4D tensor of type *T_IND* with shape ``[B, H_sel, L, K_max]``, where ``K_max = block_topk``.
  Contains only selected complete-block IDs. The valid prefix ``sel_indices[..., :sel_count]`` is **strictly
  ascending and duplicate-free**; ``-1`` is padded only after the prefix (no interior holes). The causal tail is
  not represented in this tensor; SparseAttention derives it per query from the logical position. **Required.**
  The standard form has no ``B_q``: one
  selection per query token, so the third dimension is ``L``. **Required.**

* **1**: ``sel_count``
  A 3D tensor of type *T_IND* with shape ``[B, H_sel, L]``.
  Number of selected complete-block IDs per selection head per query, satisfying
  ``0 <= sel_count <= min(block_topk, n_complete)``. Tail tokens are not counted. **Required.**

For the QSA configuration ``H_sel = 1``, flattening the batch and query-token axes gives the selection ABI
``sel_indices[T, K]`` and ``sel_count[T]``, with ``K = block_topk``. The explicit ``B`` and ``H_sel`` axes above
are retained for the generalized batched/multi-selection-head form.

The only outputs are ``sel_indices`` and ``sel_count``. The ``summary_cache``, ``indexer_raw_k_cache`` and
``indexer_rope_cache`` inputs are mutated in place; no updated cache tensors are returned.

**In-place state semantics**

* Inputs ``summary_cache``, ``indexer_raw_k_cache`` and ``indexer_rope_cache`` MUST refer to writable storage
  allocated by the caller. The
  operation reads existing valid entries and writes only new raw keys, new block-start RoPE rows and newly
  completed summaries; successful
  execution does not allocate, reserve, replace, resize, or grow any of these buffers. Existing valid entries not involved
  in the current update remain unchanged.
* The caller MUST serialize invocations that write the same cache storage. A runtime or graph optimizer MUST
  preserve the operation's side effect and its ordering relative to dependent cache reads/writes; it MUST NOT
  eliminate or duplicate the operation.
* When the cache inputs are model inputs, the bound tensors MUST be device-visible storage of the exact port
  element type (e.g. a GPU RemoteTensor or USM host tensor); a runtime that would stage them through a host copy
  or precision conversion MUST reject the binding instead of silently dropping the update.
* Before invocation, ``past_lens[b]`` raw-key entries, the first ``floor(past_lens[b] / r)`` summary entries and the
  first ``ceil(past_lens[b] / r)`` RoPE-cache rows MUST be valid. On success, the caller advances each logical
  length by ``L``. Capacity growth, rollback, and
  beam branching are managed by the caller; cache storage used by distinct active branches MUST NOT alias.
* The caller owns cache allocation and capacity growth, and MUST ensure all shape and capacity preconditions before
  invocation. If an invocation fails after
  mutation begins, the caller MUST restore or reinitialize the affected cache state before reusing it.

**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32`` (``past_lens`` included).
* ``q_idx`` and ``k_idx`` share the first two dims ``[B, L]``; both last dimensions equal ``indexer_head_dim``.
  ``q_idx`` penultimate dimension is ``H_iq`` and ``k_idx`` penultimate dimension is ``H_ik``, with
  ``H_iq % H_ik == 0`` and ``G = H_iq / H_ik`` index-query heads sharing each index-key head.
* ``H_ik`` (index-key heads) is taken from ``k_idx`` shape; ``H_sel`` is ``1`` or ``H_ik``.
* ``indexer_raw_k_cache`` dim 0 equals ``B``, dim 1 equals ``H_ik``, and dim 2 is the raw-key capacity ``S_cap``.
  ``summary_cache`` dim 0 equals ``B``, dim 1 equals ``H_ik``, and dim 2 equals
  ``floor(S_cap / compress_ratio)``. Both cache inputs MUST be writable.
* ``past_lens`` shape equals ``[B]`` and each value satisfies ``0 <= past_lens[b]`` and
  ``past_lens[b] + L <= S_cap``. ``indexer_raw_k_cache[b, :, :past_lens[b], :]`` contains the valid raw-key history;
  ``summary_cache[b, :, :floor(past_lens[b] / compress_ratio), :]`` contains the valid completed summaries.
* ``rope_cos`` and ``rope_sin`` both have shape ``[B, L, indexer_rotary_dim / 2]``, sharing ``[B, L]`` with
  ``q_idx``. All ``B x L`` tokens are valid new tokens; padding is not supported.
* ``indexer_rope_cache`` dim 0 equals ``B``, dim 1 equals ``ceil(S_cap / compress_ratio)`` (one more than
  ``summary_cache`` dim 2 when ``S_cap % compress_ratio != 0``) and dim 2 equals ``indexer_rotary_dim``; it must
  be writable and stores one row per completed-or-pending block, not a per-token or ring-capacity tensor.
* ``sel_indices`` last dimension ``K_max`` equals ``block_topk``; ``sel_count[..., sh] <= K_max`` and is at most
  ``min(block_topk, n_complete)`` for each query. The
  valid prefix ``sel_indices[..., sh, :sel_count[..., sh]]`` is **strictly ascending** and contains no ``-1``
  entries.
* For query ``l`` in batch element ``b``, the logical position is ``past_lens[b] + l``. The caches retain full
  logical history up to their valid lengths; they are not ring or pending-only tensors.
* If ``sel_count == 0``, a causal SparseAttention consumer may still produce a nonzero result from its
  automatically-derived tail. It produces zero only when no effective key remains after selection and tail
  handling.

**Types**

* *T*: any floating point type.
* *T_IND*: ``int32``.


**Example**

The example below shows one query token (``B=1``, ``L=1``) at logical position ``past_lens[0] + l = 9``, with
``past_lens[0] = 9``, ``S_cap = 16``, ``compress_ratio = 4``, and ``block_topk = 2`` (so ``K_max = 2``).
``summary_cache`` has four allocated summary slots, of which the first two are valid; ``indexer_raw_k_cache`` has
capacity for 16 keys, of which positions ``[0, 9)`` are valid. The history covers complete blocks
``{0:[0..3], 1:[4..7]}`` and an incomplete block ``2:[8]``. Causally-valid complete blocks for ``pos = 9`` are
``{0, 1}``. After a stable top-2 by score, the selection head emits ``sel_indices = [[[[0, 1]]]]`` and
``sel_count = [[[2]]]`` in strictly ascending order. The consumer independently derives the tail as logical tokens
``[8, 9]`` for this query; neither its block ID nor its token positions appear in the selection output. With
``H_iq = 4`` and ``H_ik = 1``, all four index-query heads participate in the single selection head's score.

.. code-block:: xml
   :force:

   <layer ... type="QSAIndexer">
       <data block_topk="2" h_sel="1" indexer_head_dim="128"
             indexer_rotary_dim="64" compress_ratio="4" eps="1e-6"
             scale="0.088388"/>
       <input>
           <port id="0">   <!-- q_idx: [B, L, H_iq, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>4</dim><dim>128</dim>
           </port>
           <port id="1">   <!-- k_idx: [B, L, H_ik, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>1</dim><dim>128</dim>
           </port>
           <port id="2">   <!-- rope_cos: [B, L, rotary_dim/2] -->
               <dim>1</dim><dim>1</dim><dim>32</dim>
           </port>
           <port id="3">   <!-- rope_sin: [B, L, rotary_dim/2] -->
               <dim>1</dim><dim>1</dim><dim>32</dim>
           </port>
           <port id="4">   <!-- summary_cache: [B, H_ik, floor(S_cap/r), D_idx], writable -->
               <dim>1</dim><dim>1</dim><dim>4</dim><dim>128</dim>
           </port>
           <port id="5">   <!-- indexer_raw_k_cache: [B, H_ik, S_cap, D_idx], writable -->
               <dim>1</dim><dim>1</dim><dim>16</dim><dim>128</dim>
           </port>
           <port id="6">   <!-- q_norm_weight: [D_idx] -->
               <dim>128</dim>
           </port>
           <port id="7">   <!-- k_norm_weight: [D_idx] -->
               <dim>128</dim>
           </port>
           <port id="8">   <!-- past_lens: [B] -->
               <dim>1</dim>
           </port>
           <port id="9">   <!-- indexer_rope_cache: [B, ceil(S_cap/r), rotary_dim], writable -->
               <dim>1</dim><dim>4</dim><dim>64</dim>
           </port>
       </input>
       <output>
           <port id="10" precision="I32">  <!-- sel_indices: [B, H_sel, L, K_max=block_topk] -->
               <dim>1</dim><dim>1</dim><dim>1</dim><dim>2</dim>
           </port>
           <port id="11" precision="I32">  <!-- sel_count: [B, H_sel, L] -->
               <dim>1</dim><dim>1</dim><dim>1</dim>
           </port>
       </output>
   </layer>