.. {#openvino_docs_ops_internal_QSAIndexer}

QSAIndexer
==========

.. meta::
  :description: Learn about QSAIndexer - the standard, stateful block-level indexer for Qwen Sparse Attention (QSA). It consumes projected raw query and key tokens plus writable full-history raw-key and summary caches, updates them in place, and emits complete-block selections with an optional causal-tail marker.

**Versioned name**: *QSAIndexer*

**Category**: *Internal*

**Short description**:
The *QSAIndexer* is the **model-specific plan producer** for Qwen Sparse Attention (QSA) in its
**standard, non-paged, stateful** form. It consumes externally projected indexer queries and keys, applies query
RMSNorm/RoPE, appends raw keys, generates summaries for completed blocks, and produces block selections. The
caller supplies writable raw-key and summary cache tensors; the operation updates both buffers in place and does
not return cache tensors as outputs. This makes it suitable for **single-sequence / fixed-batch**
SDPA pipelines that do not use a paged KV cache. For continuous batching over a PagedAttention page table, see the separate
:doc:`PagedQSAIndexer <paged-qsa-indexer>` operation.

**State and in-place update contract**

Unlike the paged variant, *QSAIndexer* reads **no page table** and owns no opaque runtime state. Its logical cache
state consists of two caller-owned writable tensors: ``summary_cache`` stores completed-block summaries, while
``indexer_raw_k_cache`` stores the full raw-key history. The cache dimensions describe allocated capacity; the
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

    s_{b, sh} = \frac{1}{\sqrt{D_{idx}}} \sum_{\substack{h \in [0, H_{iq}) \\ \mathrm{sel}(h) = sh}}
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

The logical cache state is stored in two preallocated, in-place updated tensors. Let :math:`S_{cap}` be the raw-key
cache capacity:

- ``summary_cache`` ``[B, H_ik, floor(S_cap / r), D_idx]`` stores mean-pooled, RMSNorm'ed, block-start-RoPE'ed
  summaries for complete blocks. For batch element ``b``, entries ``[0, floor(past_lens[b] / r))`` are valid
  before the invocation. Newly completed summaries are written into this same tensor; old valid entries are not
  recomputed during ordinary incremental execution.
- ``indexer_raw_k_cache`` ``[B, H_ik, S_cap, D_idx]`` stores all projected, unnormalized, unrotated per-token
  indexer keys. For batch element ``b``, entries ``[0, past_lens[b])`` are valid before the invocation. Current
  keys are written at their logical positions ``past_lens[b] + l``. This full-history layout uses :math:`O(S_{cap})`
  storage and permits direct reread/replay of historical raw keys.

The caller ensures capacity before invocation and advances ``past_lens`` by ``L`` after a successful call. The
precondition is ``past_lens[b] + L <= S_cap`` for every batch element; the operation does not resize either cache.
The **logical position** of query token ``l`` in batch element ``b`` is ``past_lens[b] + l``. Query RoPE uses
``position_ids``. Summary RoPE uses logical block-start position ``block_id * compress_ratio``. A block is
*completed* once exactly :math:`r` raw keys have been written. The last partially-filled block is not summarized
or selected as a complete block; its logical ID is appended as the optional final tail marker, and the paired
sparse-attention consumer processes only its visible causal prefix.

**Score computation**

For a query at logical position :math:`p` with projected query :math:`q_h \in \mathbb{R}^{D_{idx}}` (per index-query
head :math:`h`) and block :math:`b` whose per-index-key-head summary key is
:math:`\bar{k}_{b, \mathrm{kv}(h)} \in \mathbb{R}^{D_{idx}}`, the per-selection-head score is the grouped reduction
over index-query heads (see the formula above), mapping each query head to its selection head via
:math:`\mathrm{sel}(h)`. The ReLU is applied *per index-query head before reduction over heads*, matching the
QSA scoring semantics. The indexer query :math:`q` arrives projected but not normalized
or rotated. The op applies query RMSNorm (via ``q_norm_weight``) and partial RoPE using ``position_ids``. The pooled
block key undergoes RMSNorm (via ``k_norm_weight``) and partial RoPE using the rotary-table row at its logical block
start.

The RMSNorm used for the pooled block key is defined as

.. math::

    \mathrm{rms\_norm}(x, w, \mathrm{eps}) = x \cdot \mathrm{rsqrt}\!\left( \mathrm{mean}(x^2) + \mathrm{eps} \right) \cdot w

computed in ``fp32`` regardless of the input precision. Qwen's Gemma-style scale :math:`(1 + w)` is folded by the
model converter into the stored weight :math:`w' = 1 + w`, so the op consumes the already-shifted weight.

**Selection, tie-breaking, and top-k**

- A block is *causally valid* for a query at logical position :math:`p` iff its last token is at most :math:`p`,
  i.e. :math:`N_{complete}(p) = \lfloor (p + 1) / r \rfloor` blocks are eligible. Only these causally-valid
  complete blocks participate in the top-k (future / not-yet-written blocks never leak into the selection).
- The number of selected *complete* blocks per selection head is capped at ``block_topk``; the output width is
  ``K_max = block_topk + 1``. The extra slot is reserved for the query's current incomplete causal-tail block.
- **Tie-breaking and ordering**: among equal scores the *smaller block index* (earlier block) wins; the op
  computes the selection as ``argsort(-scores, kind='stable')``, takes the first ``k`` entries, and emits them in
  **strictly ascending** block order (``sort(order[:k])``), yielding a deterministic, implementation-independent,
  monotone result. The valid output prefix ``sel_indices[t, sh, :sel_count[t, sh]]`` is sorted ascending.

**Block grid and causal rules**

The KV history of a batch element is partitioned into blocks of :math:`r` tokens. Block :math:`b` covers tokens
:math:`[b \cdot r, (b+1) \cdot r - 1]`. Scoring is causal: a query at logical position :math:`p` may only select
complete blocks with :math:`b \le \lfloor (p+1)/r \rfloor - 1`. When :math:`(p+1) \bmod r \ne 0`, the output
appends the causal-tail block ID :math:`\lfloor (p+1)/r \rfloor` after the selected complete blocks. This final ID
identifies the block whose visible token prefix is consumed as the causal tail; it is not a request to attend the
whole block. The paired sparse-attention consumer must interpret this final slot as tail metadata and must not
process it again as a complete selected block. There is no ``is_causal`` attribute; the exact per-token causal
truncation is applied by the consumer via the **token-level fine filter**
:math:`kpos \le q_{abs}` (see the consumer specs).

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def qsa_indexer(q_raw, k_raw, position_ids, rotary_cos_sin, summary_cache,
            indexer_raw_k_cache, past_lens, q_norm_weight, k_norm_weight,
            *, compress_ratio, block_topk, h_sel, indexer_rotary_dim,
            indexer_head_dim, eps, scale):
      # q_raw: [B, L, H_iq, D_idx], external projection output, not normed/RoPE'd
      # k_raw: [B, L, H_ik, D_idx], external projection output, unnormed/unrotated
      # position_ids: [B, L], query RoPE table indices
      # rotary_cos_sin: [max_positions, 2 * indexer_rotary_dim]
      # summary_cache: [B, H_ik, floor(S_cap/r), D_idx], writable in place
      # indexer_raw_k_cache: [B, H_ik, S_cap, D_idx], writable in place
      # past_lens: [B], valid raw-key lengths before this invocation
      B, L, H_iq, Di = q_raw.shape
      Hik = k_raw.shape[2]
      Hsel, r = h_sel, compress_ratio
      Kmax = block_topk + 1
      G = H_iq // Hik
      S_cap = indexer_raw_k_cache.shape[2]
      assert H_iq % Hik == 0 and Hsel in (1, Hik) and r >= 1
      assert summary_cache.shape[2] == S_cap // r
      assert past_lens.shape == (B,)

      q = rms_norm(q_raw, q_norm_weight, eps)
      q = apply_partial_rope(q, rotary_cos_sin[position_ids], indexer_rotary_dim)

      sel_indices = full((B, Hsel, L, Kmax), -1, dtype=int32)
      sel_count = zeros((B, Hsel, L), dtype=int32)
      for batch in range(B):
        S0 = past_lens[batch]
        assert 0 <= S0 and S0 + L <= S_cap
        assert summary_cache.shape[2] >= (S0 + L) // r
        for l in range(L):
          indexer_raw_k_cache[batch, :, S0 + l, :] = k_raw[batch, l, :, :]

        n_complete_present = (S0 + L) // r
        for b in range(S0 // r, n_complete_present):
          grp = indexer_raw_k_cache[batch, :, b*r:(b+1)*r, :]
          pooled = mean(grp.astype(float32), axis=1)
          normed = rms_norm(pooled, k_norm_weight, eps)
          cs = rotary_cos_sin[b*r]  # logical block-start position
          summary_cache[batch, :, b, :] = apply_partial_rope(normed, cs, indexer_rotary_dim)

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
            if (pos + 1) % r != 0:
              sel_indices[batch, sh, l, k] = (pos + 1) // r
              sel_count[batch, sh, l] = k + 1
      return sel_indices, sel_count


**Attributes**

* *compress_ratio*

  * **Description**: :math:`r`, the number of raw token keys pooled into one block key.
  * **Type**: ``int``
  * **Default value**: ``4``
  * **Required**: *no*
  * **Constraints**: ``compress_ratio >= 1``.

* *block_topk*

  * **Description**: Maximum number of *complete* blocks selected per selection head. The output width is
    ``block_topk + 1``; the final slot, when present, identifies the causal-tail block.
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
    ``indexer_rotary_dim`` dims of a block summary are rotated; the trailing
    ``indexer_head_dim - indexer_rotary_dim`` dims are passed through unrotated. Must satisfy
    ``indexer_rotary_dim <= indexer_head_dim``.
  * **Type**: ``int``
  * **Required**: *yes*

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

  * **Description**: Score normalization constant, ``1.0 / sqrt(indexer_head_dim)``.
  * **Type**: ``float``
  * **Required**: *yes*


**Inputs**

* **0**: ``q_idx``
  A 4D tensor of type *T* with shape ``[B, L, H_iq, D_idx]``.
  Raw projected indexer queries before indexer RMSNorm and RoPE. The operation applies both internally. **Required.**

* **1**: ``k_idx``
  A 4D tensor of type *T* with shape ``[B, L, H_ik, D_idx]``.
  Current-step projected indexer keys. The index-key head count ``H_ik`` is derived from this shape and may be
  greater than one. Keys are appended to the
  full-history raw-K cache and pooled into newly completed-block summaries. **Required.**

* **2**: ``position_ids``
  A 2D integer tensor with shape ``[B, L]`` containing the rotary-table index for each query token. **Required.**

* **3**: ``rotary_cos_sin``
  A 2D tensor of type *T* with shape ``[max_positions, 2 * indexer_rotary_dim]``. The shared rotary table used for
  query positions and logical summary block starts. Query rows use ``position_ids``; summary rows use
  ``block_id * compress_ratio``. **Required.**

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


**Outputs**

* **0**: ``sel_indices``
  A 4D tensor of type *T_IND* with shape ``[B, H_sel, L, K_max]``.
  Selected complete-block IDs followed, when the query is inside an incomplete block, by that causal-tail block ID.
  The valid prefix ``sel_indices[..., :sel_count]`` is **strictly ascending and duplicate-free**; ``-1`` is padded
  only after the prefix (no interior holes). The final tail ID is a marker for the consumer's causal token range,
  not a complete block to attend a second time.
  The standard form has no ``B_q``: one
  selection per query token, so the third dimension is ``L``. **Required.**

* **1**: ``sel_count``
  A 3D tensor of type *T_IND* with shape ``[B, H_sel, L]``.
  Number of valid output entries (complete-block IDs plus an optional causal-tail marker) per selection head per
  query. **Required.**

The only outputs are ``sel_indices`` and ``sel_count``. The ``summary_cache`` and ``indexer_raw_k_cache`` inputs
are mutated in place; no updated cache tensors are returned.

**In-place state semantics**

* Inputs ``summary_cache`` and ``indexer_raw_k_cache`` MUST refer to writable storage allocated by the caller. The
  operation reads existing valid entries and writes only new raw keys and newly completed summaries; successful
  execution does not allocate, reserve, replace, resize, or grow either buffer. Existing valid entries not involved
  in the current update remain unchanged.
* The caller MUST serialize invocations that write the same cache storage. A runtime or graph optimizer MUST
  preserve the operation's side effect and its ordering relative to dependent cache reads/writes; it MUST NOT
  eliminate or duplicate the operation.
* Before invocation, ``past_lens[b]`` raw-key entries and the first ``floor(past_lens[b] / r)`` summary entries
  MUST be valid. On success, the caller advances each logical length by ``L``. Capacity growth, rollback, and
  beam branching are managed by the caller; cache storage used by distinct active branches MUST NOT alias.
* The caller owns cache allocation and capacity growth, and MUST ensure all shape and capacity preconditions before
  invocation. If an invocation fails after
  mutation begins, the caller MUST restore or reinitialize the affected cache state before reusing it.

**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32`` or ``int64``.
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
* ``position_ids`` shape equals ``[B, L]``; ``rotary_cos_sin`` dim 1 equals ``2 * indexer_rotary_dim`` and covers
  all query positions and completed-block start positions used by the invocation.
* ``sel_indices`` last dimension ``K_max`` equals ``block_topk + 1``; ``sel_count[..., sh] <= K_max`` and is at most
  ``min(block_topk, n_complete) + tail_present`` for each query, where ``tail_present`` is one iff
  ``(abs_pos + 1) % compress_ratio != 0``. The
  valid prefix ``sel_indices[..., sh, :sel_count[..., sh]]`` is **strictly ascending** and contains no ``-1``
  entries.
* For query ``l`` in batch element ``b``, the logical position is ``past_lens[b] + l``. The caches retain full
  logical history up to their valid lengths; they are not ring or pending-only tensors.
* When the selection for a row is empty (``sel_count == 0``), the consumer must produce a zero output vector for
  that row.

**Types**

* *T*: any floating point type.
* *T_IND*: ``int32`` or ``int64``.


**Example**

The example below shows one query token (``B=1``, ``L=1``) at logical position ``past_lens[0] + l = 9``, with
``past_lens[0] = 9``, ``S_cap = 16``, ``compress_ratio = 4``, and ``block_topk = 2`` (so ``K_max = 3``).
``summary_cache`` has four allocated summary slots, of which the first two are valid; ``indexer_raw_k_cache`` has
capacity for 16 keys, of which positions ``[0, 9)`` are valid. The history covers complete blocks
``{0:[0..3], 1:[4..7]}`` and an incomplete block ``2:[8]``. Causally-valid complete blocks for ``pos = 9`` are
``{0, 1}``. After a stable top-2 by score, the selection head emits ``sel_indices = [[[[0, 1, 2]]]]`` and
``sel_count = [[[3]]]`` in strictly ascending order. The final block ID ``2`` identifies the partial causal tail
and is not attended as a complete block. With ``H_iq = 4`` and ``H_ik = 1``, all four index-query heads
participate in the single selection head's score.

.. code-block:: xml
   :force:

   <layer ... type="QSAIndexer">
       <data block_topk="2" h_sel="1" indexer_head_dim="128"
             indexer_rotary_dim="128" compress_ratio="4" eps="1e-6"
             scale="0.088388"/>
       <input>
           <port id="0">   <!-- q_idx: [B, L, H_iq, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>4</dim><dim>128</dim>
           </port>
           <port id="1">   <!-- k_idx: [B, L, H_ik, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>1</dim><dim>128</dim>
           </port>
           <port id="2">   <!-- position_ids: [B, L] -->
               <dim>1</dim><dim>1</dim>
           </port>
           <port id="3">   <!-- rotary_cos_sin: [max_positions, 2*rotary_dim] -->
               <dim>16</dim><dim>256</dim>
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
       </input>
       <output>
             <port id="9" precision="I32">  <!-- sel_indices: [B, H_sel, L, K_max] -->
               <dim>1</dim><dim>1</dim><dim>1</dim><dim>3</dim>
           </port>
             <port id="10" precision="I32">  <!-- sel_count: [B, H_sel, L] -->
               <dim>1</dim><dim>1</dim><dim>1</dim>
           </port>
       </output>
   </layer>