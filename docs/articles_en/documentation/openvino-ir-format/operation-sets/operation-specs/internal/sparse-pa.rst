.. {#openvino_docs_ops_internal_SparsePA}

SparsePA
========

.. meta::
  :description: Learn about SparsePA - a unified, model-agnostic paged sparse attention operation that consumes a clean (selected_block_indices, selected_block_count, key_block_size) selection contract, performs online-softmax attention over selected physical KV cache blocks via an explicit page table, and in-place writes the current-step KV. Kept as an internal dev op pending fusion with PagedAttention.

**Versioned name**: *SparsePA*

**Category**: *Internal*

**Short description**:
The *SparsePA* operation performs sparse, selected-block group-query attention over a *physically paged* KV
cache, using the clean ``(selected_block_indices, selected_block_count, key_block_size)`` selection emitted by an upstream
*model-specific* producer (e.g. ``PagedQSAIndexer``). It is the paged-memory sibling of ``SparseSDPA`` and a
**model-agnostic plan consumer**. It contains **no producer-leaked artifacts**: no ``index_weights``, no output
gate, no RoPE/norm parameters, and no ``r-1`` tail hack. It is RoPE- and norm-free; positional and normalization
math lives in the main attention layer and the upstream indexer.

.. note::
   ``SparsePA`` is retained as an **independent internal development operation** — a fully self-contained,
   executable paged sparse-attention kernel. Once its contract stabilizes, the plan is to **fuse it into
   ``PagedAttention``** as an optional sparse-plan mode (XAttention-style), at which point this standalone op may
   be deprecated. Until then it is the canonical paged execution path.

**Detailed description**

*SparsePA* is the GPU-native execution form of the same sparse-attention kernel as ``SparseSDPA``, with the
addition of an **in-place KV write** of the current-step tokens. The operation:

1. Consumes the producer's clean block selection: ``selected_block_indices`` (selected *block* indices per selection
   head), ``selected_block_count`` (valid block count per head), and ``key_block_size`` (the selection block granularity).
2. **Writes the current-step ``key`` / ``value``** (the already-projected current tokens) into the paged
   ``key_cache`` / ``value_cache`` at the physical block / slot determined by the page table and ``past_lens``.
3. Resolves each selected logical block to its physical block via ``block_indices`` /
   ``block_indices_begins``.
4. Gathers the selected **complete** K/V blocks *at block granularity* (a block-gather that maps to hardware tile
    loads, rather than arbitrary per-token scatter). When ``causal = true``, it independently derives each query's
    logical position and adds the visible causal-tail token span; this span is not emitted by or counted in the
    producer's selection. **When ``causal = false``**, no causal tail is added, and the op still enforces the upper
    bound :math:`kpos < \mathrm{past\_lens}[s] + \mathrm{chunk\_len}_s`, so it never reads unallocated or stale
    slots in the physical pages.
5. Applies online-softmax attention over the selected complete blocks and, when causal, the automatically-derived
    tail tokens.
6. Writes the per-head output.

**Dimension naming: ``total_q_blocks`` vs ``tokens``**

The producer emits the selection with a **block-grouped first dimension** ``total_q_blocks``. The token stream is
partitioned into :math:`B_q`-token query blocks per sequence; the total number of query blocks across all sequences
is

.. math::

    \mathrm{total\_q\_blocks} = \sum_{s} \left\lceil \frac{L_s}{B_q} \right\rceil

where :math:`L_s = \mathrm{subsequence\_begins}[s{+}1] - \mathrm{subsequence\_begins}[s]` is the number of
current-step tokens of sequence ``s``. When :math:`B_q = 1` (the common QSA token-sparse decode case),
``total_q_blocks == tokens``. When :math:`B_q > 1` (e.g. FlashVSR block sparse with :math:`B_q = 128`), each
query block carries one selection replicated across its rows, and ``total_q_blocks`` is the number of such
blocks. **Only ``selected_block_indices``/``selected_block_count`` are indexed by ``total_q_blocks``**; the ``query``/``output`` tensors
remain per-token (``[tokens, ...]``), because a query block's selection is applied to every token inside it. The
selection for the query block covering sequence ``s``, token ``l`` is at row
``qb_start[s] + (l - subsequence_begins[s]) // B_q``, where ``qb_start`` is the running prefix sum of the
per-sequence query-block counts (an optional input, equal to ``subsequence_begins`` when :math:`B_q = 1`). When :math:`B_q > 1`, the history length of each sequence must be block-aligned:
``past_lens[s] % query_block_size == 0``.

**Paged memory organization**

The KV cache is organized into non-contiguous physical blocks of fixed ``page_size`` tokens (the paged block size
of the main attention cache). ``page_size`` is not an attribute: as in *PagedAttentionExtension*, it is derived
from the cache shape, ``page_size = key_cache.shape[2]``. For sequence ``s``, the assigned physical block indices are
``block_indices[block_indices_begins[s] : block_indices_begins[s+1]]``; the logical-to-physical map is
identical to the one used by ordinary PagedAttention (``PagedAttention`` / ``PaKVReorder``), so a single
scheduler owns all paged memory. The physical KV block size ``page_size`` is **decoupled** from the producer's
selection block size ``key_block_size``: only the divisibility constraint ``page_size % key_block_size == 0``
is required, so one selected block maps to a *contiguous slice* of a physical block without forcing the
selection granularity to equal the paging granularity. When the producer shares the page table (as
``PagedQSAIndexer`` does), the page geometry must coincide with the physical cache: its ``page_size`` equals ``key_cache.shape[2]``,
and the producer's ``block_index_keys_cache`` has the same first dimension ``num_blocks`` as ``key_cache``.

**Block granularity and the clean contract**

The producer emits block-level selection at granularity ``key_block_size``. For QSA, the plan contains only
selected *complete* blocks, each loaded whole (all ``key_block_size`` tokens); it never carries a causal-tail ID.
For every causal query, SparsePA computes
:math:`tail\_start = \lfloor (token\_pos + 1) / B_k \rfloor B_k` and additionally gathers logical token
positions :math:`tail\_start, \ldots, token\_pos` inclusive. The span is empty when the query is at a complete
block boundary. The plan is a **strictly ascending, duplicate-free** sequence of complete-block indices: the op
gathers exactly the blocks named in the contiguous prefix ``selected_block_indices[:selected_block_count]`` and does not infer or
modify that selection. The causal tail is an independent, mandatory attention span, not a selection entry. An
empty complete-block selection can still produce a nonzero result from its causal tail; output is zero only when
there are no effective keys after applying plan, causal, and validity bounds.

The same rule applies when a plan is supplied as ``selected_block_mask``: in causal mode it describes selected complete
blocks only. The bit for the query's incomplete tail block is not a producer selection; SparsePA adds the visible
tail tokens independently for each query.

**Mask-only plans for unbounded-K_max producers**

Not every producer has a fixed per-row top-:math:`K`. Threshold-based selection — e.g. XAttention's block
scoring, which greedily accumulates the highest-scoring blocks until their *cumulative softmax mass* crosses a
threshold fraction, with no cap on how many blocks that takes — is **data-dependent**: for a near-uniform score
distribution the loop keeps going until almost every block of the sequence is selected, so the worst-case
selected-block count approaches that sequence's full logical block grid. Unlike QSA's fixed ``block_topk``
complete-block bound,
there is no tighter bound to declare, and forcing such a plan through ``selected_block_indices`` then pays for a full-width
``int32`` index per logical block per query row — in the worst case carrying no more information than "every
block of this sequence is selected" — instead of the 1 bit a boolean mask would need.

``SparsePA`` has no token-level ``attn_mask`` input to reuse for this (it operates on a ragged, physically paged
token stream rather than a dense ``[B, H, L, S]`` layout), so the mask-only alternative here is a **block-level**
boolean mask, ``selected_block_mask``, carried **in place of** the ``(selected_block_indices, selected_block_count)`` pair. When a producer
has no bound on ``K_max`` tighter than a sequence's full logical block count, it supplies
``selected_block_mask[row, sh, :num_k_blocks_s]`` directly — one bit per logical selection block of that row's
sequence, where ``num_k_blocks_s = (block_indices_begins[s+1] - block_indices_begins[s]) * (page_size /
key_block_size)`` — instead of an ascending index list, and ``selected_block_indices``/``selected_block_count`` are passed as empty tensors. This
mirrors the GPU implementation of XAttention (``xattn_find_block.cm`` / ``xattn_post_proc.cm``), which computes
exactly a per-(head, q-block, k-block) boolean mask and never converts it into an index list. It also gives the
standalone dev op an equivalent of the representation its eventual fusion target would use: see "Relationship to
``PagedAttention`` and ``XAttention``" below, where XAttention's threshold selection is a mask-mode sparse plan of
the fused ``PagedAttention`` primitive rather than an index list.

One of the two plan representations must be present: either the ``selected_block_indices``/``selected_block_count`` pair, or
``selected_block_mask`` with ``selected_block_indices``/``selected_block_count`` both empty tensors (input ports are positional and cannot be
skipped).

**Causal responsibility and the slot-read upper bound**

Causality is split across the producer and the consumer:

- **Producer (e.g. ``PagedQSAIndexer``) — complete-block plan only.** The producer selects only complete,
  causally-valid blocks (block :math:`b` eligible iff :math:`(b{+}1) \cdot B_k - 1 \le pos`). It does not emit
  the causal diagonal or tail block, and the tail is not included in ``selected_block_count``.
- **Consumer (this op) — token-level fine filter (Token-level Precision).** The consumer is the final authority
  on token-level causality: it gathers the selected complete blocks, derives and adds the per-query tail span
  :math:`[\lfloor (token\_pos+1)/B_k \rfloor B_k, token\_pos]` (inclusive), and imposes
  :math:`kpos \le token\_pos`. This guarantees causal attention regardless of producer granularity or selection
  width.

When :math:`causal = \mathrm{false}`, there is no lower truncation, but the op must **still** guard the *upper*
bound: a key at logical position :math:`kpos` is only valid if it has actually been written for this sequence,
i.e. :math:`kpos < \mathrm{past\_lens}[s] + \mathrm{chunk\_len}_s`, where
:math:`\mathrm{chunk\_len}_s = \mathrm{subsequence\_begins}[s{+}1] - \mathrm{subsequence\_begins}[s]` is the
number of current-step tokens of sequence ``s`` (so the newest written logical position is
:math:`\mathrm{past\_lens}[s] + \mathrm{chunk\_len}_s - 1`). This prevents the non-causal gather from reading
**unallocated or stale slots** in the physical pages (e.g. pages that were reused by the scheduler or never
written), which would otherwise feed garbage into the softmax. Under ``causal = true`` this upper bound is
implied by the tighter :math:`kpos \le token\_pos` (the diagonal), but it is applied unconditionally for safety.

This division mirrors ``SparseSDPA``: the producer performs the coarse block filter and the consumer the fine
token filter.

**Relationship to ``PagedAttention`` and ``XAttention``**

``SparsePA`` generalizes ordinary paged attention in two independent axes — sparsity (attending only a subset
of blocks, per the selection contract) and block-granular selection (``key_block_size`` may be finer than the
physical ``page_size``). When the selection covers every block (:math:`H_{sel}=1` and ``selected_block_count`` equals the
full block count) and ``key_block_size == page_size``, the kernel degenerates to plain paged causal attention.
This mirrors the XAttention design, in which sparsity is a selectable execution mode of the paged-attention
primitive rather than a separate operator. The intended final form is therefore to absorb ``SparsePA`` into
``PagedAttention`` as an optional sparse-plan input group (mutually exclusive with the XAttention threshold
path); ``SparsePA`` remains a standalone internal dev op until that fusion lands. In the meantime, a
threshold-based producer (no fixed top-:math:`K`) that targets this op directly should carry its plan via
``selected_block_mask`` rather than ``selected_block_indices``/``selected_block_count`` — see "Mask-only plans for unbounded-K_max
producers" above.

.. note::
   **Known limitations vs. full ``PagedAttention``** — as a standalone dev op, ``SparsePA`` does *not* yet
   provide the full feature set of ``PagedAttention``: no sink / sliding-window blocks, no quantization-aware
   cache, no ALiBi-style positional bias, and no copy-on-write page sharing. It assumes a single, statically
   allocated physical block list per sequence. These gaps are the primary drivers for the planned fusion into
   ``PagedAttention`` (as an optional sparse-plan mode), where the full PA feature set applies.

**Head mapping**

Selection heads :math:`H_{sel} \in \{1, H_{kv}, H_{q}\}` with :math:`H_{q} \bmod H_{sel} == 0`. Query head
:math:`h` maps to selection head :math:`sh = \lfloor h / (H_{q} / H_{sel}) \rfloor`. When :math:`H_{sel}=1`
(QSA), all query heads share the single selection; when :math:`H_{sel}=H_{kv}`, each group of
:math:`H_{q} / H_{kv}` query heads shares a per-KV-head selection; when :math:`H_{sel}=H_{q}`, every query head
carries its own selection.

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def sparse_pa(query, key, value, key_cache, value_cache,
                  past_lens, subsequence_begins, block_indices, block_indices_begins,
                  selected_block_indices, selected_block_count, scale, sinks, qb_start=None,
                  *, num_heads, num_kv_heads, num_selection_heads, k_head_size, v_head_size,
                  key_block_size, query_block_size=1, causal):
        # query:            [tokens, num_heads * k_head_size]  (projected/RMSNorm'ed/RoPE'ed; per token)
        # key, value:       [tokens, num_kv_heads * D]          (current-step tokens, in-place written)
        # key_cache:        [num_blocks, num_kv_heads, page_size, k_head_size]
        # value_cache:      [num_blocks, num_kv_heads, page_size, v_head_size]
        # past_lens:        [num_sequences]  (logical KV length per sequence before this step)
        # subsequence_begins:   [num_sequences + 1]  split pointers into the token stream
        # block_indices:    [total_logical_blocks]   logical -> physical block map
        # block_indices_begins: [num_sequences + 1]  split pointers into block_indices
        # selected_block_indices:      [total_q_blocks, H_sel, K_max]  (int32; strictly ascending, -1 padded past selected_block_count)
        # selected_block_count:        [total_q_blocks, H_sel]         (int32; valid complete-block count)
        # qb_start:         optional [num_sequences + 1]  running prefix sum of per-sequence query-block counts
        #                   (qb_start[s] = sum_{s'<s} ceil(chunk_len_{s'} / Bq); qb_start[-1] == total_q_blocks);
        #                   None when query_block_size == 1, where it equals subsequence_begins
        # scale:            scalar attention scale, typically 1 / sqrt(k_head_size)
        # sinks:            [num_heads] or empty; a non-empty sinks adds one extra logit per query head to the
        #                   softmax denominator (not supported by the GPU implementation)
        tokens = query.shape[0]
        if qb_start is None:
            qb_start = subsequence_begins              # query_block_size == 1
        Hq, Hkv = num_heads, num_kv_heads
        Dh, Dv = k_head_size, v_head_size
        B = key_cache.shape[2]                        # page_size, derived from the cache
        Bk = key_block_size
        Bq = query_block_size
        blk_per_page = B // Bk                        # selection blocks per physical page
        q_per_sel = Hq // num_selection_heads                       # query heads per selection head (Hq % num_selection_heads == 0)

        q = reshape(query, (tokens, Hq, Dh))
        out = zeros((tokens, Hq, Dv))

        for s in range(num_sequences):
            chunk_len = subsequence_begins[s + 1] - subsequence_begins[s]
            valid_max = past_lens[s] + chunk_len      # exclusive upper bound on valid logical kpos
            phys_seq = block_indices[block_indices_begins[s] : block_indices_begins[s + 1]]
            # In-place KV write of the current chunk: physical block/slot from the shared page table.
            for t in range(subsequence_begins[s], subsequence_begins[s + 1]):
                token_pos = past_lens[s] + (t - subsequence_begins[s])   # global logical position
                phys_block = phys_seq[token_pos // B]
                offset_in_block = token_pos % B
                key_cache[phys_block, :, offset_in_block, :] = key[t]
                value_cache[phys_block, :, offset_in_block, :] = value[t]

            # Attention over the selected blocks, applied per token. Each token t of sequence s belongs to the
            # query block at row = qb_start[s] + (t - subsequence_begins[s]) // Bq.
            for t in range(subsequence_begins[s], subsequence_begins[s + 1]):
                row = qb_start[s] + (t - subsequence_begins[s]) // Bq      # query-block row in selected_block_indices
                token_pos = past_lens[s] + (t - subsequence_begins[s])     # global logical position
                for h in range(Hq):
                    sh = h // q_per_sel                   # selection head of query head h
                    kv = h // (Hq // Hkv)                 # KV head of query head h
                    nblocks = selected_block_count[row, sh]
                    idx = selected_block_indices[row, sh, :nblocks]          # valid logical block indices (ascending set)
                    # Map each selected selection-block to its physical block + in-block selection offset.
                    phys = phys_seq[idx // blk_per_page]          # physical block ids [nblocks]
                    offs = (idx % blk_per_page) * Bk              # in-block selection offset [nblocks]
                    # Gather selected complete blocks at key_block_size granularity. The absolute KV position
                    # of each gathered token is derived from its logical block index: kpos = idx*Bk + off.
                    K_g, V_g = gather_physical_blocks(key_cache, value_cache,
                                                      phys, offs, Bk)   # [n_sel*Bk, Hkv, Dh], [n_sel*Bk, Hkv, Dv]
                    kpos = concat([idx[:, None] * Bk + arange(Bk)[None, :]], axis=-1).reshape(-1)  # abs pos
                    # Causal tail is implicit and independent of the producer's complete-block selection.
                    # Derive it for this token (not once per query block); it is empty at a block boundary.
                    if causal:
                        tail_start = ((token_pos + 1) // Bk) * Bk
                        tail_pos = arange(tail_start, token_pos + 1)
                        K_tail, V_tail = gather_logical_positions(
                            key_cache, value_cache, phys_seq, tail_pos, block_size=B)
                        K_g = concat([K_g, K_tail], axis=0)
                        V_g = concat([V_g, V_tail], axis=0)
                        kpos = concat([kpos, tail_pos], axis=0)
                    # Token-level fine filter. Lower bound: keep keys at positions <= the query's logical
                    # position (self-inclusive) when causal. Upper bound (always): never read a slot that has
                    # not been written for this sequence (valid_max), so causal=false never touches unallocated
                    # or stale slots in the physical pages.
                    eff = ones(kpos.shape[0], dtype=bool)
                    eff = eff & (kpos < valid_max)                 # never read unwritten/stale slots
                    if causal:
                        eff = eff & (kpos <= token_pos)            # causal truncation, self-inclusive
                    scores = sum(q[t, h] * K_g[:, kv, :], axis=-1) * scale   # [n_sel*Bk]
                    scores = where(eff, scores, -inf)
                    probs = softmax(scores, axis=-1)               # online, over valid tokens only
                    if not np.any(eff):
                        probs = zeros_like(probs)                  # fully-masked row -> zero (avoid NaN)
                    out[t, h] = sum(probs[:, None] * V_g[:, kv, :], axis=0)  # [Dv]

        return out                                            # [tokens, Hq*Dv]


**Attributes**

* *num_heads*

  * **Description**: Number of query heads :math:`H_q`.
  * **Type**: ``int``
  * **Required**: *yes*

* *num_kv_heads*

  * **Description**: Number of KV heads :math:`H_{kv}`. Must divide ``num_heads``.
  * **Type**: ``int``
  * **Required**: *yes*

* *num_selection_heads*

  * **Description**: Number of selection heads :math:`H_{sel}`, matching the producer. Must be ``1``,
    ``num_kv_heads``, or ``num_heads``.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``num_selection_heads`` is ``1``, ``num_kv_heads``, or ``num_heads``; ``num_heads % num_selection_heads == 0``.

* *k_head_size*

  * **Description**: Head dimension :math:`D_h` of the keys.
  * **Type**: ``int``
  * **Required**: *yes*

* *v_head_size*

  * **Description**: Head dimension :math:`D_v` of the values.
  * **Type**: ``int``
  * **Required**: *yes*

* *key_block_size*

  * **Description**: Number of tokens per selected block, :math:`B_k`, matching the producer's
    ``key_block_size``. Must divide ``page_size`` (``key_cache.shape[2]``). A selected block is a contiguous slice
    of a physical block.
  * **Type**: ``int``
  * **Required**: *yes*

* *query_block_size*

  * **Description**: Number of query tokens sharing a single selection, :math:`B_q`. Default ``1`` (decode /
    per-token selection, so ``total_q_blocks == tokens``). When :math:`B_q > 1` (e.g. FlashVSR block sparse with
    :math:`B_q = 128`), the first selection dimension is
    :math:`\mathrm{total\_q\_blocks} = \sum_s \lceil L_s / B_q \rceil`, and the history length of each sequence
    must be block-aligned: ``past_lens[s] % query_block_size == 0``.
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*

* *causal*

  * **Description**: Whether the consumer automatically adds each query's causal-tail tokens, with valid keys
    satisfying :math:`kpos \le token\_pos` (inclusive, self-attended). The tail is derived from the logical query
    position and is not supplied by the producer. When ``false``, all written tokens of every selected block are
    attended — but the op still enforces the upper bound
    :math:`kpos < \mathrm{past\_lens}[s] + \mathrm{chunk\_len}_s` so it never reads unallocated or stale slots
    in the physical pages.
  * **Type**: ``bool``
  * **Default value**: ``true``
  * **Required**: *no*


**Inputs**

* **0**: ``query``
  A 2D tensor of type *T* with shape ``[tokens, num_heads * k_head_size]``.
  Current-step query hidden states, already projected, RMSNorm'ed and RoPE'ed by the attention layer, one row per
  token. **Required.**

* **1**: ``key``
  A 2D tensor of type *T* with shape ``[tokens, num_kv_heads * k_head_size]``.
  Current-step projected keys, written in place into ``key_cache``. **Required.**

* **2**: ``value``
  A 2D tensor of type *T* with shape ``[tokens, num_kv_heads * v_head_size]``.
  Current-step projected values, written in place into ``value_cache``. **Required.**

* **3**: ``key_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, num_kv_heads, page_size, k_head_size]``; ``page_size`` is read
  from ``shape[2]``. Physically paged key cache; read for scoring and written in place. **Required.**

* **4**: ``value_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, num_kv_heads, page_size, v_head_size]``.
  Physically paged value cache; read for gathering and written in place. **Required.**

* **5**: ``past_lens``
  A 1D tensor of type *T_IND* with shape ``[num_sequences]``.
  Logical KV length of each sequence before this step. The global logical position of a token is
  ``past_lens[s] + (t - subsequence_begins[s])``; it determines the in-place write slot, the causal truncation
  bound, and the upper bound on valid logical positions for non-causal reads. **Required.**

* **6**: ``subsequence_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Splits the token stream among sequences. Sequence ``s`` uses tokens
  ``[subsequence_begins[s] : subsequence_begins[s + 1]]``. **Required.**

* **7**: ``block_indices``
  A 1D tensor of type *T_IND* with shape ``[total_logical_blocks]``.
  Physical block indices for all sequences; ``total_logical_blocks = block_indices_begins[-1]``.
  **Required.**

* **8**: ``block_indices_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Splits ``block_indices`` among sequences. Sequence ``s`` uses blocks
  ``block_indices[block_indices_begins[s] : block_indices_begins[s+1]]``. **Required.**

* **9**: ``selected_block_indices``
  A 3D tensor of type *T_IND* with shape ``[total_q_blocks, num_selection_heads, K_max]``.
  Per-query-block, per-selection-head selected **complete-block** indices emitted by the upstream producer, as a
  **strictly ascending, duplicate-free** sequence in the contiguous prefix ``selected_block_indices[:selected_block_count]``. In causal
  mode, the plan must not contain the current incomplete tail block; SparsePA derives that tail itself. ``-1``
  padded only past ``selected_block_count``. When ``query_block_size == 1``, the first dimension ``total_q_blocks`` equals
  ``tokens``. **Required**, unless the plan is instead carried by ``selected_block_mask`` (see "Mask-only plans for
  unbounded-K_max producers" above), in which case this input is an empty tensor.

* **10**: ``selected_block_count``
  A 2D tensor of type *T_IND* with shape ``[total_q_blocks, num_selection_heads]``.
  Number of selected complete-block IDs per selection head; it excludes any automatically-derived causal tail.
  With ``causal = true``, ``selected_block_count == 0`` can still yield a nonzero output from the tail. **Required, unless
  the plan is carried by** ``selected_block_mask``, in which case this input is an empty tensor.

* **11**: ``scale``
  A scalar tensor of type *T*: the attention scale factor, typically ``1 / sqrt(k_head_size)``. The GPU
  implementation requires it to be a ``Constant``. **Required.**

* **12**: ``sinks``
  A 1D tensor of type *T_sink* with shape ``[num_heads]``, or an empty tensor: the attention sink logit of every
  query head. An empty tensor means no sinks. **Reserved:** a non-empty ``sinks`` is not supported by the current
  implementation yet (connecting it is an error). **Required (empty tensor = no sinks).**

* **13**: ``qb_start``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Running prefix sum of the per-sequence query-block counts: ``qb_start[0] == 0``,
  ``qb_start[s] == sum_{s'<s} ceil(chunk_len_{s'} / query_block_size)``, and
  ``qb_start[-1] == total_q_blocks``. Maps a (sequence, token) to the row of its query block in
  ``selected_block_indices``/``selected_block_count`` (or ``selected_block_mask``) ``/query/output``.
  **Optional:** when ``query_block_size == 1`` every token is its own query block, so ``qb_start`` equals
  ``subsequence_begins`` and the input is omitted. It is required when ``query_block_size > 1``, and also when
  ``selected_block_mask`` (port 14) is provided, because ports are positional; an empty tensor then stands for "absent".

* **14**: ``selected_block_mask``
  A 3D boolean tensor with shape ``[total_q_blocks, num_selection_heads, max_k_blocks]``, where ``max_k_blocks`` is the
  largest per-sequence logical block count in the batch, :math:`\max_s ((\mathrm{block\_indices\_begins}[s{+}1] -
  \mathrm{block\_indices\_begins}[s]) \cdot \mathrm{page\_size} / \mathrm{key\_block\_size})`. For a row
  belonging to sequence ``s``, only the first ``num_k_blocks_s`` entries are meaningful; entries past that are
  padding. **Optional** — the last input, used in place of ``selected_block_indices``/``selected_block_count`` for
  producers with no fixed per-row top-:math:`K` (see "Mask-only plans for unbounded-K_max producers" above). It is
  omitted (the op then has 13 or 14 inputs) or an empty tensor when ``selected_block_indices``/``selected_block_count``
  carry the plan. Providing it requires port 13 (``qb_start``) to be present, or an empty tensor. The current
  implementation does not support it: connecting a non-empty mask is an error.

.. note::
   Input ports 0-12 are positional and cannot be skipped, as for *PagedAttentionExtension*; use an empty tensor
   for an unused ``sinks`` (and for ``selected_block_indices``/``selected_block_count`` in the mask-only form).
   ``qb_start`` (13) and ``selected_block_mask`` (14) are optional trailing inputs.


**Outputs**

* **0**: ``output``
  A 2D tensor of type *T* with shape ``[tokens, num_heads * v_head_size]``.
  Sparse attention output, concatenated across heads, one row per token.


**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32``.
* ``num_heads % num_kv_heads == 0``.
* ``num_selection_heads`` is ``1``, ``num_kv_heads``, or ``num_heads``; ``num_heads % num_selection_heads == 0``.
* ``key_cache``/``value_cache`` second dimension equals ``num_kv_heads``; their third dimension is ``page_size``
  (not an attribute) and both caches share it.
* ``page_size % key_block_size == 0``; the producer's ``key_block_size`` must divide the physical ``page_size``.
  When the producer shares the page table, its ``page_size`` equals this one and its summary cache has the same
  first dimension ``num_blocks``.
* ``scale`` is a scalar or single-element tensor. ``sinks`` is either empty or has shape ``[num_heads]``.
* ``block_indices`` values must be valid physical block indices ``< num_blocks``.
* ``query`` and ``output`` first dimension equals ``tokens``.
* ``selected_block_indices``/``selected_block_count`` (or ``selected_block_mask``) first dimension equals ``total_q_blocks``, where
  ``total_q_blocks == qb_start[-1] == sum_s ceil(chunk_len_s / query_block_size)``; when ``query_block_size == 1`` this
  equals ``tokens``.
* ``selected_block_indices`` last dimension ``K_max`` is **determined by the producer**: ``block_topk`` for a QSA-style
  producer (the causal tail is implicit and not counted), or ``ceil(S / key_block_size)`` when the selection is
  produced from a block-level mask conversion.
  When a producer has no bound on ``K_max`` tighter than a sequence's full logical block count (e.g.
  XAttention's threshold-based selection), ``selected_block_indices``/``selected_block_count`` should be empty and the plan carried
  directly via ``selected_block_mask`` instead — see "Mask-only plans for unbounded-K_max producers".
* ``selected_block_count[row, sh] <= K_max`` for every row ``row`` and selection head ``sh``; the valid prefix
  ``selected_block_indices[row, sh, :selected_block_count[row, sh]]`` is **strictly ascending** and contains no ``-1`` entries.
* Exactly one plan representation is present: either both ``selected_block_indices`` and ``selected_block_count``, or
  ``selected_block_mask`` with ``selected_block_indices``/``selected_block_count`` both empty. ``selected_block_mask``'s last dimension,
  ``max_k_blocks``, equals :math:`\max_s` of each sequence's logical block count
  ``(block_indices_begins[s+1] - block_indices_begins[s]) * page_size / key_block_size``; for a row of sequence
  ``s``, entries past that sequence's own logical block count are padding and not read.
* ``block_indices_begins[0] == 0`` and ``block_indices_begins[-1] == len(block_indices)`` (block splits);
  ``subsequence_begins[0] == 0`` and ``subsequence_begins[-1] == tokens`` (token splits).
* ``past_lens[s]`` equals the number of tokens already written for sequence ``s``; the write slot
  ``past_lens[s] + (t - subsequence_begins[s])`` must lie within an allocated physical block.
* When ``query_block_size > 1``, ``past_lens[s] % query_block_size == 0`` for every sequence ``s``.
* When ``causal`` is true, the op automatically adds the per-query causal-tail positions from
  :math:`\lfloor (token\_pos+1)/key\_block\_size \rfloor key\_block\_size` through ``token_pos`` inclusive;
  this tail is not part of ``selected_block_indices`` or ``selected_block_count``. The op also enforces :math:`kpos \le token\_pos` for
  all gathered keys. It **always** enforces the upper bound
  :math:`kpos < \mathrm{past\_lens}[s] + \mathrm{chunk\_len}_s`, so a non-causal gather never reads unallocated
  or stale slots in the physical pages.
* If ``selected_block_count == 0`` and ``causal`` is true, the automatically-added causal tail can still produce a nonzero
  output. The output is zero only when no effective key remains after plan, tail, causal, and validity filtering.


**Types**

* *T*: any floating point type.
* *T_sink*: any floating point type; it may differ from *T*.
* *T_IND*: ``int32``.


**Examples**

**Example 1 — Qwen4 Token Sparse** (:math:`B_q = 1`, :math:`B_k = 4`, :math:`H_{sel} = 1`).

A batch of two sequences, each with a fully-paged KV cache, attending with ``key_block_size = 4`` (so
``K_max = block_topk = 2`` complete blocks; the causal tail is added by SparsePA), ``page_size = 8`` (a multiple of
``key_block_size = 4``), ``num_heads = 24``, ``num_kv_heads = 2``, ``num_selection_heads = 1``. With ``B_q = 1``, the first
selection dimension ``total_q_blocks == tokens``; the example shows ``tokens = 4`` (e.g. two sequences of two
tokens each, ``chunk_len = [2, 2]``; ``qb_start = [0, 2, 4]`` equals ``subsequence_begins`` and is omitted). ``selected_block_indices`` encodes, per query token, only
the top-``K`` complete blocks. When ``causal = true``, SparsePA independently derives and adds each token's tail
up to :math:`kpos \le token\_pos`.

.. code-block:: xml
   :force:

   <layer ... type="SparsePA">
       <data num_heads="24" num_kv_heads="2" num_selection_heads="1" k_head_size="256" v_head_size="256"
             key_block_size="4" query_block_size="1" causal="true"/>
       <input>
           <port id="0">  <!-- query: [tokens, Hq*Dh] -->
               <dim>4</dim><dim>6144</dim>
           </port>
           <port id="1">  <!-- key: [tokens, Hkv*Dh] -->
               <dim>4</dim><dim>512</dim>
           </port>
           <port id="2">  <!-- value: [tokens, Hkv*Dv] -->
               <dim>4</dim><dim>512</dim>
           </port>
           <port id="3">  <!-- key_cache: [num_blocks, Hkv, page_size, Dh] -->
               <dim>8</dim><dim>2</dim><dim>8</dim><dim>256</dim>
           </port>
           <port id="4">  <!-- value_cache: [num_blocks, Hkv, page_size, Dv] -->
               <dim>8</dim><dim>2</dim><dim>8</dim><dim>256</dim>
           </port>
           <port id="5">  <!-- past_lens: [num_sequences] -->
               <dim>2</dim>
           </port>
           <port id="6">  <!-- subsequence_begins: [num_sequences+1] -->
               <dim>3</dim>
           </port>
           <port id="7">  <!-- block_indices: [total_logical_blocks] -->
               <dim>6</dim>
           </port>
           <port id="8">  <!-- block_indices_begins: [num_sequences+1] -->
               <dim>3</dim>
           </port>
           <port id="9">  <!-- selected_block_indices: [total_q_blocks, H_sel, K_max] (== tokens when Bq == 1) -->
               <dim>4</dim><dim>1</dim><dim>2</dim>
           </port>
           <port id="10">   <!-- selected_block_count: [total_q_blocks, H_sel] -->
               <dim>4</dim><dim>1</dim>
           </port>
           <port id="11">   <!-- scale: scalar -->
               
           </port>
           <port id="12">   <!-- sinks: [num_heads], empty = no sinks -->
               <dim>0</dim>
           </port>
       </input>
       <output>
           <port id="13">   <!-- output: [tokens, Hq*Dv] -->
               <dim>4</dim><dim>6144</dim>
           </port>
       </output>
   </layer>

**Example 2 — FlashVSR Block Sparse** (:math:`B_q = 128`, :math:`B_k = 128`, :math:`H_{sel} = H_q = H_{kv}`,
non-causal).

FlashVSR's temporal block sparsity is expressed as a block-level sparse mask carried once per :math:`128`-token
query block, using **multi-head attention** (:math:`H_q = H_{kv} = H_{sel} = 12`). With one sequence of
``tokens = 512`` and :math:`B_q = 128`, the first selection dimension is
``total_q_blocks = ceil(512 / 128) = 4`` and ``qb_start = [0, 4]``. Each of the 4 query blocks carries its own
:math:`H_{sel} = H_q` selection over the :math:`S`-token KV history. A block-level sparse mask that keeps blocks
``{0, 2, 3}`` for every selection head of query block ``qb=0`` is encoded as
``selected_block_indices[0, :, :] = [[0, 2, 3, -1, ...], ...]`` (replicated across the :math:`H_{sel}` selection heads) and
``selected_block_count[0, :] = [3, 3, ...]``. With :math:`B_k = 128` and :math:`S = 512` there are ``S / B_k = 4`` logical
blocks; the selection covers ``3`` of them. Non-causal (:math:`causal = \mathrm{false}`), so all *written* tokens
of every selected block are attended — the op still applies the upper bound
:math:`kpos < \mathrm{past\_lens}[s] + \mathrm{chunk\_len}_s` (here ``0 + 512 = 512``), which excludes the
out-of-range tail of the final partial block and never touches unallocated/stale slots. **The input tokens must be
pre-permuted into window-major order** (FlashVSR ``WindowPartition3D``) so that each :math:`B_q`/:math:`B_k` block
is exactly one 3D window.

.. code-block:: xml
   :force:

   <layer ... type="SparsePA">
       <data num_heads="12" num_kv_heads="12" num_selection_heads="12" k_head_size="128" v_head_size="128"
             key_block_size="128" query_block_size="128" causal="false"/>
       <input>
           <port id="0">  <!-- query: [tokens, Hq*Dh] -->
               <dim>512</dim><dim>1536</dim>
           </port>
           <port id="1">  <!-- key: [tokens, Hkv*Dh] -->
               <dim>512</dim><dim>1536</dim>
           </port>
           <port id="2">  <!-- value: [tokens, Hkv*Dv] -->
               <dim>512</dim><dim>1536</dim>
           </port>
           <port id="3">  <!-- key_cache: [num_blocks, Hkv, page_size, Dh] -->
               <dim>4</dim><dim>12</dim><dim>128</dim><dim>128</dim>
           </port>
           <port id="4">  <!-- value_cache: [num_blocks, Hkv, page_size, Dv] -->
               <dim>4</dim><dim>12</dim><dim>128</dim><dim>128</dim>
           </port>
           <port id="5">  <!-- past_lens: [num_sequences] -->
               <dim>1</dim>
           </port>
           <port id="6">  <!-- subsequence_begins: [num_sequences+1] -->
               <dim>2</dim>
           </port>
           <port id="7">  <!-- block_indices: [total_logical_blocks] -->
               <dim>4</dim>
           </port>
           <port id="8">  <!-- block_indices_begins: [num_sequences+1] -->
               <dim>2</dim>
           </port>
           <port id="9">  <!-- selected_block_indices: [total_q_blocks, H_sel, K_max] -->
               <dim>4</dim><dim>12</dim><dim>4</dim>
           </port>
           <port id="10">   <!-- selected_block_count: [total_q_blocks, H_sel] -->
               <dim>4</dim><dim>12</dim>
           </port>
           <port id="11">   <!-- scale: scalar -->
               
           </port>
           <port id="12">   <!-- sinks: [num_heads], empty = no sinks -->
               <dim>0</dim>
           </port>
           <port id="13">   <!-- qb_start: [num_sequences+1] -->
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="14">   <!-- output: [tokens, Hq*Dv] -->
               <dim>512</dim><dim>1536</dim>
           </port>
       </output>
   </layer>

**Example 3 — XAttention Threshold-Based Sparsity** (:math:`B_q = B_k = 128`, plan carried by
``selected_block_mask``, ``selected_block_indices``/``selected_block_count`` empty).

XAttention's block selection accumulates the highest-scoring blocks per query block until their cumulative
softmax mass crosses a threshold fraction, so the selected-block count is content-dependent with no fixed cap.
Reusing the one-sequence, ``tokens = 512``, :math:`B_q = B_k = 128` layout of Example 2
(``total_q_blocks = 4``, ``qb_start = [0, 4]``, one physically paged sequence with ``num_k_blocks_s = 4``
logical blocks, so ``max_k_blocks = 4``): suppose query block ``qb=0`` keeps blocks ``{0, 2, 3}`` (as in
Example 2) while query block ``qb=3`` (near-uniform scores) keeps all of ``{0, 1, 2, 3}`` — a fixed-width
``selected_block_indices`` would need ``K_max = 4 == num_k_blocks_s`` to cover the worst case, buying no index-list savings
for ``qb=3``. The plan is instead supplied as the boolean ``selected_block_mask``
(``selected_block_mask[0, :, :] = [[1, 0, 1, 1], ...]``, ``selected_block_mask[3, :, :] = [[1, 1, 1, 1], ...]``, replicated
across the :math:`H_{sel}` selection heads); ``selected_block_indices`` and ``selected_block_count`` are empty tensors.

.. code-block:: xml
   :force:

   <layer ... type="SparsePA">
       <data num_heads="12" num_kv_heads="12" num_selection_heads="12" k_head_size="128" v_head_size="128"
             key_block_size="128" query_block_size="128" causal="false"/>
       <input>
           <port id="0">  <!-- query: [tokens, Hq*Dh] -->
               <dim>512</dim><dim>1536</dim>
           </port>
           <port id="1">  <!-- key: [tokens, Hkv*Dh] -->
               <dim>512</dim><dim>1536</dim>
           </port>
           <port id="2">  <!-- value: [tokens, Hkv*Dv] -->
               <dim>512</dim><dim>1536</dim>
           </port>
           <port id="3">  <!-- key_cache: [num_blocks, Hkv, page_size, Dh] -->
               <dim>4</dim><dim>12</dim><dim>128</dim><dim>128</dim>
           </port>
           <port id="4">  <!-- value_cache: [num_blocks, Hkv, page_size, Dv] -->
               <dim>4</dim><dim>12</dim><dim>128</dim><dim>128</dim>
           </port>
           <port id="5">  <!-- past_lens: [num_sequences] -->
               <dim>1</dim>
           </port>
           <port id="6">  <!-- subsequence_begins: [num_sequences+1] -->
               <dim>2</dim>
           </port>
           <port id="7">  <!-- block_indices: [total_logical_blocks] -->
               <dim>4</dim>
           </port>
           <port id="8">  <!-- block_indices_begins: [num_sequences+1] -->
               <dim>2</dim>
           </port>
           <port id="9">  <!-- selected_block_indices: empty (plan carried by selected_block_mask) -->
               <dim>0</dim><dim>12</dim><dim>0</dim>
           </port>
           <port id="10">   <!-- selected_block_count: empty (plan carried by selected_block_mask) -->
               <dim>0</dim><dim>12</dim>
           </port>
           <port id="11">   <!-- scale: scalar -->
               
           </port>
           <port id="12">   <!-- sinks: [num_heads], empty = no sinks -->
               <dim>0</dim>
           </port>
           <port id="13">   <!-- qb_start: [num_sequences+1] -->
               <dim>2</dim>
           </port>
           <port id="14">   <!-- selected_block_mask: [total_q_blocks, H_sel, max_k_blocks] -->
               <dim>4</dim><dim>12</dim><dim>4</dim>
           </port>
       </input>
       <output>
           <port id="15">   <!-- output: [tokens, Hq*Dv] -->
               <dim>512</dim><dim>1536</dim>
           </port>
       </output>
   </layer>
