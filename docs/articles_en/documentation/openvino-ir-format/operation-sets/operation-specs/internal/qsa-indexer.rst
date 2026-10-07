.. {#openvino_docs_ops_internal_QSAIndexer}

QSAIndexer
==========

.. meta::
  :description: Learn about QSAIndexer - a model-specific, weight-free block-level indexer for Qwen Sparse Attention that consumes externally projected query and token keys, maintains an incremental pooled-key summary cache over the main PagedAttention page table plus a per-sequence pending ring buffer, and emits a clean (sel_indices, sel_count) block selection contract for the unified SparseSDPA / SparsePA consumers.

**Versioned name**: *QSAIndexer*

**Category**: *Internal*

**Short description**:
The *QSAIndexer* is the **model-specific plan producer** for Qwen Sparse Attention (QSA). It consumes an
already-projected indexer query ``q_idx`` and per-token projected keys ``k_idx`` (produced by an ordinary
``FullyConnected`` / ``MatMul`` outside the op), then performs the *K-side compression chain* — per-token or
pooled key compression (mean-pool), GemmaRMSNorm, and RoPE at the historical block start — maintains the
incremental ``summary_cache``, scores each block with a per-head ReLU dot-product reduced across *index* heads,
and selects the top-:math:`K` blocks. It emits a **clean, producer-complete** ``(sel_indices, sel_count)`` block
selection contract. The unified consumers ``SparseSDPA`` / ``SparsePA`` are deliberately *plan consumers*:
they only gather KV at the selected blocks and perform online-softmax attention. No ``index_weights``, no
output gate, no RoPE/norm parameters, and no ``r-1`` tail hack leak from the producer into the consumer.

**Detailed description**

The *QSAIndexer* realizes the *producer* side of the sparse-attention split. It is *model-specific*: it
encodes the exact QSA reference math (mean-pool :math:`\to` RMSNorm :math:`\to` block-start RoPE :math:`\to`
per-head ReLU score :math:`\to` top-k), in contrast to the *unified* ``SparseSDPA`` / ``SparsePA`` consumers
which are model-agnostic. The indexer GEMM weight (``index_qk_proj``) is **kept outside** this op as a
standard ``FullyConnected`` / ``MatMul`` for four reasons:

1. **Quantization (NNCF INT4/FP8).** Keeping the projection as an ordinary MatMul lets the NNCF weight-
   compression passes rewrite it without touching an opaque fused primitive. Dequantizing a compressed
   projection to BF16 for an in-op GEMM costs orders of magnitude more DRAM traffic than the <0.1% decode
   traffic the fusion would save.
2. **Horizontal FC fusion.** A standalone projection can be fused horizontally with the main
   ``q_proj``/``k_proj``/``v_proj`` for the shared input, a transform impossible when the GEMM is embedded in
   the indexer primitive.
3. **Tensor parallelism (TP).** A standalone MatMul can be sharded across devices by head / dimension without
   fighting a fused kernel's fixed input layout.
4. **Kernel combinatorics.** Fusing the projection into the indexer multiplies the kernel family by every
   precision / layout combination; keeping it a standard FC keeps the kernel space small.

The *K-side compression chain*, by contrast, **must** stay inside the indexer semantics because it is
*non-associative* and *incremental*: :math:`\mathrm{RMSNorm}(\mathrm{Mean}(K)) \ne \mathrm{Mean}(\mathrm{RMSNorm}(K))`,
and a block summary can only be maintained :math:`O(1)` per new token by an internal cache. Weight-free-ness
therefore applies to *weights* (explicit, external GEMM) and not to *compression semantics* (atomic, cached,
inside the op).

**Data flow and cache organization**

The K-side history is maintained as two caches that reuse the main PagedAttention page table and add one small
per-sequence ring buffer:

- ``summary_cache`` ``[num_blocks, 1, page_size / compress_ratio, D_idx]``: the mean-pooled, RMSNorm'ed,
  block-start-RoPE'ed summary key of each *completed* block. This is the incrementally-updated artifact that
  makes the per-token update :math:`O(1)`. It is addressed by the **same page table** as the main PagedAttention
  KV cache (``block_indices`` / ``block_indices_begins``). For sequence ``s``, block ``b`` (covering logical
  tokens ``[b*r, (b+1)*r - 1]``) is stored at the physical location
  ``summary_cache[phys, 0, slot_in_block, :]`` with
  ``phys = block_indices[block_indices_begins[s] + (b * r) // page_size]`` and
  ``slot_in_block = (b * r % page_size) // r``. This requires ``page_size % compress_ratio == 0`` so every
  summary slot maps to a fixed position inside one physical page.
- ``pending_k_cache`` ``[num_slots, compress_ratio, 1, D_idx]``: the projected, per-token indexer keys of the
  *currently incomplete* block of each sequence, held in a **per-sequence ring buffer** (one ring slot per
  sequence, addressed by ``slot_mapping``). It is addressed **by logical position**: for a token at logical
  position :math:`pos`, the ring offset is :math:`off = pos \bmod r` and the key is stored in
  ``pending_k_cache[slot, off, 0, :]``. Position addressing makes the op **idempotent and decode-continuous**:
  re-running a step writes the same slot, and a chunked prefill that resumes at ``off == 0`` re-establishes the
  same block-start state without any internal counter. The ring is *not* a full-history cache (dropping the
  legacy ``raw_k_cache`` avoids its +12.5% VRAM overhead). RoPE is not baked into these raw keys.

  .. note::
     **Decode mode.** With ``capacity = compress_ratio`` (exactly one block per slot), the ring supports
     **standard greedy / autoregressive decoding only**. Re-running an arbitrary recent step to **roll back a
     speculative-decode draft** is *not* supported: the ring is overwritten in place once a block completes and
     would lose the tokens needed to replay a rolled-back prefix. To support rollback of a speculative draft of
     length :math:`N`, the ring capacity would need to be :math:`C \ge r + N` (as done by vLLM), retaining the
     last :math:`N` tokens of the previous block alongside the current incomplete block. This spec adopts the
     compact :math:`C = r` design and defers that extension.
- ``pending_start_cos_sin`` ``[num_slots, 2 * indexer_rotary_dim]``: the precomputed RoPE ``(cos, sin)`` at the
  block-start position of the pending block. It is refreshed at the start of each block, i.e. when the first
  token of a block (``off == 0``) arrives: ``pending_start_cos_sin[slot] = rope_cos_sin[t]`` (see the
  pseudo-code).
- ``rope_cos_sin`` ``[tokens, 2 * indexer_rotary_dim]``: per-token RoPE ``(cos, sin)`` for the query step,
  materialized from the full table by the caller (decouples logical index from RoPE coordinate, see below).

A block is *completed* once exactly :math:`r` raw keys have been appended (the pending ring fills and its
summary is flushed into ``summary_cache``). The last, partially-filled block of a sequence is *not* summarized
into ``summary_cache``; instead it is emitted as a **valid block index** in the output (the producer-completeness
rule), so the consumer gathers it as a partial block without any ``r-1`` tail hack in the selection encoding.

**Logical index vs. RoPE coordinate (M-RoPE)**

The op separates the *logical KV index* used for block addressing and causal filtering from the *RoPE
coordinate* used for positional rotation. The **logical position** of a token is derived from
``past_lens`` and the sequence partition (``subsequence_begins``): for sequence ``s`` and token ``t``,
:math:`pos = past\_lens[s] + (t - subsequence\_begins[s])`. This logical position drives (a) which blocks are
causally valid and (b) the pending ring slot addressing. The **RoPE coordinate** for the key side is a purely
positional quantity, captured once per block in ``pending_start_cos_sin``; for the query side it is supplied per
token via ``rope_cos_sin``, which the caller materializes from the full table using the appropriate (possibly
M-RoPE) coordinate. This decoupling is what makes 3-axis M-RoPE expressible: each axis rotates over its own
coordinate (token index, image row, image column) taken from the appropriate position tensor, while the block
grid and page-table addressing stay purely logical.

**Completeness of the emitted selection**

The emitted plan is a **set of block indices** — unordered and duplicate-free. Every block a query must attend
to is present as a *valid* index in the contiguous prefix ``sel_indices[:sel_count]``:

- the **top-k scored** complete blocks among the causally-valid complete blocks,
- the **causal diagonal / incomplete block** (unconditionally appended immediately after the selected complete
  blocks, so the consumer knows to gather it partially).

No sink / sliding-window declarations are made here: any such blocks, if required by a model, are scored and
selected through the same top-k path, and the causal-diagonal block is always included. The valid indices are
written contiguously from slot ``0`` (no interior ``-1`` holes); only the tail past ``sel_count`` is ``-1``
padded. This is the *producer-completeness* contract: the emitted plan is a **block-level coarse filter (Block
Eligibility)** — it guarantees no future / not-yet-written complete block leaks into the selection, and it is
position-free. The consumers (``SparseSDPA`` / ``SparsePA``) never re-derive the block selection; they gather
exactly the blocks named by the valid prefix of ``sel_indices`` and counted by ``sel_count``, and apply the
**token-level fine filter (Token-level Precision)** — the exact :math:`kpos \le q_{abs}` causal truncation that
is the final guarantee of correctness. An empty selection (``sel_count == 0``, e.g. a query with no
causally-valid complete block) yields a zero-filled selection; the consumer produces a zero output vector for
that row.

**Score computation**

For a query at logical position :math:`p` with projected query :math:`q \in \mathbb{R}^{H_{idx}\times D_{idx}}`
and block :math:`b` whose (single) summary key is :math:`\bar{k}_b \in \mathbb{R}^{D_{idx}}`, the per-selection-
head score is a **grouped reduction over index heads**:

.. math::

    s_{b, sh} = \frac{1}{\sqrt{D_{idx}}} \sum_{h \in \mathrm{group}(sh)} \mathrm{ReLU}\!\left( \langle \mathrm{RoPE}_{pos(p)}(\mathrm{RMSNorm}(q_h)),\, \bar{k}_b \rangle \right)

where :math:`\bar{k}_b = \mathrm{RoPE}_{b\cdot r}(\mathrm{RMSNorm}(\tfrac{1}{r}\sum_{j=0}^{r-1} k_{b\cdot r + j}))`
(a single key head). The ReLU is applied *per index head before reduction over heads*, matching the reference QSA
implementation. The indexer query :math:`q` arrives already projected (external GEMM); the op applies the query
RMSNorm and partial RoPE from ``q_norm_weight`` and ``rope_cos_sin`` at the query's own position.

The reduction is over ``num_index_heads`` (:math:`H_{idx}`) index heads, partitioned into **disjoint groups**
indexed by the selection head :math:`sh \in [0, H_{sel})`. Each group :math:`\mathrm{group}(sh)` is the set of
index heads that contribute to selection head :math:`sh`; the groups must be a partition of :math:`[0, H_{idx})`.
Under QSA (:math:`H_{sel}=1``) the single group contains all :math:`H_{idx}` index heads, so the score is reduced
over every index head and emitted on the single selection head. When :math:`H_{sel}>1` the group size is
:math:`H_{idx} / H_{sel}`, which requires :math:`H_{idx} \bmod H_{sel} == 0`. Because the key summary
:math:`\bar{k}_b` is a single vector (there is only one key head), every index head in a group scores against
the same :math:`\bar{k}_b`; only the query head differs.

**Selection heads vs. index heads**

The op distinguishes two head counts:

- ``num_index_heads`` (:math:`H_{idx}`): the heads of the indexer projection. Each head produces an independent
  query/key summary, the per-head ReLU dot-product is summed (in groups) over these heads to yield one scalar per
  selection head per block.
- ``h_sel`` (:math:`H_{sel}`): the heads of the emitted *selection*. Under QSA, :math:`H_{sel} = 1` — a single
  shared selection drives every query/KV head of the main attention. Models with per-KV-head selection use
  :math:`H_{sel} = H_{kv}` of the main attention.

Under QSA the score is reduced over all :math:`H_{idx}` index heads (a single group) and then broadcast to the
single :math:`H_{sel}=1` selection; the two counts coincide only when the model picks a single index head. The
grouping constraint is :math:`H_{idx} \bmod H_{sel} == 0` (see "Score computation").

**Selection, tie-breaking, and top-k**

- A block is *causally valid* for a query at logical position :math:`p` iff its last token is at most :math:`p`,
  i.e. the number of complete blocks a query may attend to is dynamic:
  :math:`N_{complete}(p) = \lfloor (p + 1) / r \rfloor`. Only these causally-valid complete blocks participate
  in the top-k (future / not-yet-written blocks never leak into the selection).
- The number of selected *complete* blocks per selection head is capped at ``block_topk``; the output width is
  ``K_max = block_topk + 1`` (the extra slot holds the causal-diagonal / incomplete block, when present). The
  top-k is taken over the ``min(block_topk, N_complete(p))`` causally-valid complete blocks and written to the
  contiguous prefix ``sel_indices[0:k]``.
- **Tie-breaking**: scores are reduced by selection-head group and normalized once. Ties in ``topk`` are broken
  by a **fixed, well-defined rule inside the op**: among equal scores, the *smaller block index* (earlier block)
  wins. Formally the op computes the selection as ``argsort(-scores, kind='stable')`` and takes the first ``k``
  entries, which yields a deterministic, implementation-independent result for any input. Consistency with the
  reference implementation is exact only when scores are strictly unequal (no ties).
- **Causal-diagonal inclusion**: when the query at logical position :math:`p` lies inside an incomplete block
  (i.e. :math:`(p+1) \bmod r \neq 0`), the diagonal block with logical index :math:`inc = \lfloor (p+1)/r
  \rfloor` is written **immediately after the selected complete blocks**, at ``sel_indices[k]``, and
  ``sel_count = k + 1``. It is always present in the selection, regardless of whether the top-k budget is
  exhausted. Writing it at ``sel_indices[k]`` (rather than a fixed slot) guarantees the valid prefix is
  contiguous with no interior ``-1`` holes, so the consumer can iterate ``sel_indices[:sel_count]`` directly.

**Block grid and causal rules**

The KV history of a sequence is partitioned into blocks of :math:`r` tokens. Block :math:`b` covers tokens
:math:`[b \cdot r, (b+1) \cdot r - 1]`. Scoring is causal: a query at logical position :math:`p` may only select
complete blocks with :math:`(b{+}1) \cdot r - 1 \le p`, i.e. :math:`b \le \lfloor (p+1)/r \rfloor - 1`; the
current incomplete block (the causal diagonal) is always appended separately. The op therefore performs the
**block-level coarse filter (Block Eligibility)**: it never emits future / not-yet-written complete blocks.
There is no ``is_causal`` attribute. The exact per-token causal truncation is *not* the producer's job — it is
applied by the consumers (``SparseSDPA`` / ``SparsePA``) via the **token-level fine filter**
:math:`kpos \le q_{abs}` (see the consumer specs).

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def qsa_indexer(q_idx, k_idx, summary_cache, pending_k_cache, pending_start_cos_sin,
                    slot_mapping, block_indices, block_indices_begins,
                    subsequence_begins, past_lens,
                    q_norm_weight, k_norm_weight, rope_cos_sin,
                    *, compress_ratio, budget, block_topk, num_index_heads, h_sel,
                    indexer_head_dim, indexer_rotary_dim, num_sequences, page_size, eps, scale):
        # q_idx:            [tokens, num_index_heads * D_idx]   (projected by external GEMM)
        # k_idx:            [tokens, D_idx]                      (projected by external GEMM)
        # summary_cache:    [num_blocks, 1, page_size // r, D_idx]
        # pending_k_cache:  [num_slots, r, 1, D_idx]             (per-sequence ring, one slot per seq)
        # pending_start_cos_sin: [num_slots, 2 * indexer_rotary_dim]
        # slot_mapping:     [num_sequences]   (seq -> ring slot)
        # block_indices:    [total_logical_blocks]   logical -> physical block map (shared with main PA)
        # block_indices_begins: [num_sequences + 1]  split pointers into block_indices (len == total_logical_blocks)
        # subsequence_begins:   [num_sequences + 1]  split pointers into the token stream (len == tokens)
        # past_lens:        [num_sequences]  (logical length of each sequence before this step)
        # rope_cos_sin:     [tokens, 2 * indexer_rotary_dim]     (per-token materialized RoPE)
        # returns: sel_indices [tokens, H_sel, K_max]  (int32; -1 = pad), K_max = block_topk + 1
        #          sel_count   [tokens, H_sel]         (int32)
        tokens = q_idx.shape[0]
        Hidx, Di = num_index_heads, indexer_head_dim
        Hsel = h_sel
        r = compress_ratio
        Kmax = block_topk + 1                       # +1 slot for the causal-diagonal / incomplete block
        head_groups = Hidx // Hsel                   # index heads per selection-head group (Hidx % Hsel == 0)
        assert page_size % r == 0 and Hidx % Hsel == 0

        # 1. Query RMSNorm + partial RoPE at the query's own position.
        #    RoPE rotates only the first `indexer_rotary_dim` dims; the tail is kept as-is.
        q = reshape(q_idx, (tokens, Hidx, Di))
        q = rms_norm(q, q_norm_weight, eps)                                  # [tokens, Hidx, Di]
        q_r = rope(q[..., :indexer_rotary_dim], rope_cos_sin)                # rotate rotary prefix
        q = concat([q_r, q[..., indexer_rotary_dim:]], axis=-1)              # keep unrotated tail

        # 2. Per-token K-side update, addressed by logical position (idempotent / decode-continuous).
        #    off = pos % r selects the ring slot; off == 0 re-records the block-start RoPE phase;
        #    off == r-1 flushes the completed block into summary_cache via the shared page table.
        for s in range(num_sequences):
            slot = slot_mapping[s]
            base = past_lens[s]                         # logical position of the first token of this chunk
            for t in range(subsequence_begins[s], subsequence_begins[s + 1]):
                pos = base + (t - subsequence_begins[s])    # global logical position
                off = pos % r
                if off == 0:
                    # first token of a block: record the block-start RoPE phase from this token's table row
                    pending_start_cos_sin[slot] = rope_cos_sin[t]
                pending_k_cache[slot, off, 0, :] = k_idx[t]
                if off == r - 1:
                    # block `b = pos // r` is now complete: mean-pool (fp32), RMSNorm, partial RoPE, flush
                    b = pos // r
                    pooled = mean(pending_k_cache[slot].astype(float32), axis=0).astype(k_idx.dtype)
                    nb = rms_norm(pooled, k_norm_weight, eps)                    # [Di]
                    bar = concat([rope(nb[..., :indexer_rotary_dim],
                                       pending_start_cos_sin[slot]),
                                  nb[..., indexer_rotary_dim:]], axis=-1)        # keep unrotated tail
                    phys = block_indices[block_indices_begins[s] + (b * r) // page_size]
                    slot_in_block = (b * r % page_size) // r
                    summary_cache[phys, 0, slot_in_block, :] = bar

        # 3. Score the causally-valid complete blocks per selection head (grouped index-head reduction).
        #    For a query at logical position `pos`, only blocks b < (pos + 1) // r are valid;
        #    future blocks never enter the top-k.
        #    seq_of_token(t) := the unique s with subsequence_begins[s] <= t < subsequence_begins[s+1].
        scores = {}
        for t in range(tokens):
            s = seq_of_token(t, subsequence_begins)
            pos = past_lens[s] + (t - subsequence_begins[s])
            n_complete = (pos + 1) // r                    # dynamic per-query complete-block count
            for sh in range(Hsel):
                h0 = sh * head_groups
                for b in range(n_complete):
                    phys = block_indices[block_indices_begins[s] + (b * r) // page_size]
                    slot_in_block = (b * r % page_size) // r
                    bar = summary_cache[phys, 0, slot_in_block, :]        # [Di], single key head
                    # grouped per-index-head ReLU reduction -> scalar
                    dot = sum(relu(sum(q[t, h0:h0 + head_groups, :] * bar[None, :], axis=-1)), axis=0)
                    scores[(t, sh, b)] = dot * scale

        # 4. Top-k over causally-valid complete blocks per selection head; stable tie-break (smaller block wins).
        sel_indices = full((tokens, Hsel, Kmax), -1, dtype=int32)
        sel_count   = zeros((tokens, Hsel), dtype=int32)
        for t in range(tokens):
            s = seq_of_token(t, subsequence_begins)
            pos = past_lens[s] + (t - subsequence_begins[s])
            n_complete = (pos + 1) // r
            for sh in range(Hsel):
                k = min(block_topk, n_complete)            # pick up to block_topk complete blocks
                scores_arr = np.array([scores[(t, sh, b)] for b in range(n_complete)])  # [n_complete]
                order = np.argsort(-scores_arr, kind='stable')
                picked = order[:k]
                sel_indices[t, sh, :k] = picked
                sel_count[t, sh] = k
                # Causal diagonal: when the query lies inside an incomplete block, append it right after
                # the selected complete blocks (slot `k`) so the valid prefix stays contiguous (no -1 holes).
                if (pos + 1) % r != 0:
                    inc = (pos + 1) // r                   # incomplete block index == floor(pos / r)
                    sel_indices[t, sh, k] = inc
                    sel_count[t, sh] = k + 1
        return sel_indices, sel_count


**Attributes**

* *compress_ratio*

  * **Description**: :math:`r`, the number of raw token keys pooled into one block key.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``compress_ratio >= 1``. ``page_size % compress_ratio == 0``.

* *budget*

  * **Description**: :math:`B`, the maximum number of selected tokens per query head across complete blocks.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``budget % compress_ratio == 0``.

* *block_topk*

  * **Description**: Maximum number of *complete* blocks selected per selection head,
    ``== budget // compress_ratio``. The output width is ``block_topk + 1`` (the extra slot is the causal-
    diagonal / incomplete block).
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``block_topk >= 1``.

* *num_index_heads* (``indexer_heads``)

  * **Description**: Number of index heads :math:`H_{idx}` that participate in the per-head ReLU score and its
    reduction over heads. Independent of ``h_sel``.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``num_index_heads >= 1``; ``q_idx`` last dim ``== num_index_heads * indexer_head_dim``.

* *h_sel*

  * **Description**: Number of selection heads :math:`H_{sel}` in the emitted contract. Must be ``1`` or the
    ``num_kv_heads`` of the main attention. The :math:`H_{idx}` index heads are partitioned into ``h_sel``
    disjoint groups (``H_idx % H_sel == 0``); each selection head's score is the sum over its group.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``h_sel >= 1``; ``h_sel == 1`` (QSA default) or ``h_sel == main num_kv_heads``;
    ``num_index_heads % h_sel == 0``.

* *indexer_head_dim*

  * **Description**: Head dimension :math:`D_{idx}` of the indexer.
  * **Type**: ``int``
  * **Required**: *yes*

* *indexer_rotary_dim*

  * **Description**: RoPE dimension used by the indexer (partial rotary). Only the first
    ``indexer_rotary_dim`` dims are rotated; the trailing ``indexer_head_dim - indexer_rotary_dim`` dims are
    passed through unrotated. Must satisfy ``indexer_rotary_dim <= indexer_head_dim``.
  * **Type**: ``int``
  * **Required**: *yes*

* *page_size*

  * **Description**: Number of tokens per physical page of the shared main PagedAttention cache. Must be a
    multiple of ``compress_ratio``.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``page_size % compress_ratio == 0``.

* *k_norm_type*

  * **Description**: Normalization applied to the pooled block key. Must be ``rms`` (RMSNorm / GemmaRMSNorm).
  * **Type**: ``enum``
  * **Required**: *yes*
  * **Supported values**: ``rms``

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
  A 2D tensor of type *T* with shape ``[tokens, num_index_heads * D_idx]``.
  Projected indexer queries emitted by the external ``FullyConnected`` / ``MatMul`` (``index_qk_proj``).
  **Required.**

* **1**: ``k_idx``
  A 2D tensor of type *T* with shape ``[tokens, D_idx]``.
  Current-step projected indexer keys, emitted by the same external ``index_qk_proj``. Stored position-addressed
  in the owning sequence's pending ring. **Required.**

* **2**: ``summary_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, 1, page_size // compress_ratio, D_idx]``.
  Mean-pooled + RMSNorm'ed + block-start-RoPE'ed summary key of each completed block, addressed through the
  shared main PA page table (``block_indices`` / ``block_indices_begins``). Updated in place. **Required.**

* **3**: ``pending_k_cache``
  A 4D tensor of type *T* with shape ``[num_slots, compress_ratio, 1, D_idx]``.
  Per-sequence ring buffer holding the projected, unrotated keys of the currently incomplete block, addressed by
  logical position (``off = pos % compress_ratio``). **Required.**

* **4**: ``pending_start_cos_sin``
  A 2D tensor of type *T* with shape ``[num_slots, 2 * indexer_rotary_dim]``.
  RoPE ``(cos, sin)`` at the block-start position of each pending ring slot. Refreshed at the first token of each
  block (``off == 0``). **Required.**

* **5**: ``slot_mapping``
  A 1D tensor of type *T_IND* with shape ``[num_sequences]``.
  Maps each sequence to its pending ring slot. **Required.**

* **6**: ``block_indices``
  A 1D tensor of type *T_IND* with shape ``[total_logical_blocks]``.
  The main PagedAttention logical-to-physical block map; also addresses ``summary_cache``. **Required.**

* **7**: ``block_indices_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Splits ``block_indices`` among sequences: sequence ``s`` owns blocks
  ``block_indices[block_indices_begins[s] : block_indices_begins[s+1]]``. **Required.**

* **8**: ``subsequence_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Splits the token stream among sequences: sequence ``s`` uses tokens
  ``[subsequence_begins[s] : subsequence_begins[s + 1]]``. **Required.**

* **9**: ``past_lens``
  A 1D tensor of type *T_IND* with shape ``[num_sequences]``.
  Logical KV length of each sequence before this step. The global logical position of a token is
  ``past_lens[s] + (t - subsequence_begins[s])``. **Required.**

* **10**: ``q_norm_weight``
  A 1D tensor of type *T* with shape ``[D_idx]``.
  RMSNorm weight for indexer queries. **Required.**

* **11**: ``k_norm_weight``
  A 1D tensor of type *T* with shape ``[D_idx]``.
  RMSNorm weight for pooled block keys. **Required.**

* **12**: ``rope_cos_sin``
  A 2D tensor of type *T* with shape ``[tokens, 2 * indexer_rotary_dim]``.
  Per-token RoPE ``(cos, sin)`` for the query step, prepared by the caller from the full table using the query
  RoPE coordinates (which may be M-RoPE). The ``(cos, sin)`` pairs are laid out in the reference
  ``rotate_half`` convention; the first ``indexer_rotary_dim`` dims are rotated and the trailing
  ``indexer_head_dim - indexer_rotary_dim`` dims are passed through unrotated. This decouples the logical index
  from the RoPE coordinate. **Required.**


**Outputs**

* **0**: ``sel_indices``
  A 3D tensor of type *T_IND* with shape ``[tokens, H_sel, block_topk + 1]``.
  Selected *block* indices per selection head as an **unordered, duplicate-free set** in the contiguous prefix
  ``sel_indices[:sel_count]``; the causal-diagonal / incomplete block (when present) is appended immediately
  after the selected complete blocks. ``-1`` padded only past ``sel_count`` (no interior holes). **Required.**

* **1**: ``sel_count``
  A 2D tensor of type *T_IND* with shape ``[tokens, H_sel]``.
  Number of valid selected blocks per selection head per query. **Required.**


**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32`` or ``int64``.
* ``tokens``, ``num_sequences`` and ``subsequence_begins`` must be consistent:
  ``subsequence_begins[0] == 0`` and ``subsequence_begins[-1] == tokens``.
* ``block_indices_begins[0] == 0`` and ``block_indices_begins[-1] == len(block_indices)`` (block splits).
* ``q_idx`` last dimension equals ``num_index_heads * indexer_head_dim``; ``k_idx`` last dimension equals
  ``indexer_head_dim``.
* ``page_size % compress_ratio == 0``; ``summary_cache`` dim 2 equals ``page_size // compress_ratio`` and dim 0
  equals ``num_blocks`` (shared with the main KV ``key_cache`` dim 0).
* ``slot_mapping`` length equals ``num_sequences``; ``pending_k_cache`` dim 0 equals ``num_slots`` and dim 1
  equals ``compress_ratio``.
* ``num_index_heads % h_sel == 0`` (index heads partition cleanly into selection-head groups).
* ``sel_indices`` last dimension is fixed at ``block_topk + 1``; ``sel_count[t, sh] <= block_topk + 1``, and the
  valid prefix ``sel_indices[t, sh, :sel_count[t, sh]]`` contains no ``-1`` entries.
* When the selection for a row is empty (``sel_count == 0``), the consumer must produce a zero output vector for
  that row.


**Types**

* *T*: any floating point type.
* *T_IND*: ``int32`` or ``int64``.


**Example**

The example below shows a single query token at logical position ``9`` (``past_lens[0] = 9``, one token in the
step) with ``compress_ratio = 4``, ``budget = 8``, ``block_topk = 2`` (so ``K_max = block_topk + 1 = 3``),
``num_index_heads = 1``, ``h_sel = 1``, ``page_size = 16``. The history written before this step covers blocks
``{0:[0..3], 1:[4..7]}`` (complete) and block ``2:[8]`` (the causal diagonal / incomplete block). Causally-valid
complete blocks for ``pos = 9`` are those with ``b < (9+1)//4 = 2``, i.e. ``{0, 1}``. After a stable top-2 by
score over ``{0, 1}`` the selection head emits the top-2 complete blocks ``{1, 0}`` plus the causal-diagonal
block ``2`` written immediately after them (producer-completeness, contiguous prefix), yielding
``sel_indices = [[[1, 0, 2]]]`` and ``sel_count = [[3]]``.

.. code-block:: xml
   :force:

   <layer ... type="QSAIndexer">
       <data compress_ratio="4" budget="8" block_topk="2"
             num_index_heads="1" h_sel="1" indexer_head_dim="128"
             indexer_rotary_dim="128" page_size="16" k_norm_type="rms" eps="1e-6"
             scale="0.088388"/>
       <input>
           <port id="0">   <!-- q_idx: [tokens, num_index_heads*D_idx] -->
               <dim>1</dim><dim>128</dim>
           </port>
           <port id="1">   <!-- k_idx: [tokens, D_idx] -->
               <dim>1</dim><dim>128</dim>
           </port>
           <port id="2">   <!-- summary_cache: [num_blocks, 1, page_size//r, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>4</dim><dim>128</dim>
           </port>
           <port id="3">   <!-- pending_k_cache: [num_slots, r, 1, D_idx] -->
               <dim>1</dim><dim>4</dim><dim>1</dim><dim>128</dim>
           </port>
           <port id="4">   <!-- pending_start_cos_sin: [num_slots, 2*rotary_dim] -->
               <dim>1</dim><dim>256</dim>
           </port>
           <port id="5">   <!-- slot_mapping: [num_sequences] -->
               <dim>1</dim>
           </port>
           <port id="6">   <!-- block_indices: [total_logical_blocks] -->
               <dim>1</dim>
           </port>
           <port id="7">   <!-- block_indices_begins: [num_sequences+1] -->
               <dim>2</dim>
           </port>
           <port id="8">   <!-- subsequence_begins: [num_sequences+1] -->
               <dim>2</dim>
           </port>
           <port id="9">   <!-- past_lens: [num_sequences] -->
               <dim>1</dim>
           </port>
           <port id="10">  <!-- q_norm_weight: [D_idx] -->
               <dim>128</dim>
           </port>
           <port id="11">  <!-- k_norm_weight: [D_idx] -->
               <dim>128</dim>
           </port>
           <port id="12">  <!-- rope_cos_sin: [tokens, 2*rotary_dim] -->
               <dim>1</dim><dim>256</dim>
           </port>
       </input>
       <output>
           <port id="13" precision="I32">  <!-- sel_indices: [tokens, H_sel, block_topk+1] -->
               <dim>1</dim><dim>1</dim><dim>3</dim>
           </port>
           <port id="14" precision="I32">  <!-- sel_count: [tokens, H_sel] -->
               <dim>1</dim><dim>1</dim>
           </port>
       </output>
   </layer>