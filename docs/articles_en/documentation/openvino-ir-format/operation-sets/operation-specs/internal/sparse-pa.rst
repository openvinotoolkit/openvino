.. {#openvino_docs_ops_internal_SparsePA}

SparsePA
========

.. meta::
  :description: Learn about SparsePA - a unified, model-agnostic paged sparse attention operation that consumes a clean (sel_indices, sel_count, sel_block_size) selection contract, performs online-softmax attention over selected physical KV cache blocks via an explicit page table, and in-place writes the current-step KV. Kept as an internal dev op pending fusion with PagedAttention.

**Versioned name**: *SparsePA*

**Category**: *Internal*

**Short description**:
The *SparsePA* operation performs sparse, selected-block group-query attention over a *physically paged* KV
cache, using the clean ``(sel_indices, sel_count, sel_block_size)`` selection emitted by an upstream
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

1. Consumes the producer's clean block selection: ``sel_indices`` (selected *block* indices per selection
   head), ``sel_count`` (valid block count per head), and ``sel_block_size`` (the selection block granularity).
2. **Writes the current-step ``key`` / ``value``** (the already-projected current tokens) into the paged
   ``key_cache`` / ``value_cache`` at the physical block / slot determined by the page table and ``past_lens``.
3. Resolves each selected logical block to its physical block via ``block_indices`` /
   ``block_indices_begins``.
4. Gathers the selected K/V token states *at block granularity* (a block-gather that maps to hardware tile
   loads, rather than arbitrary per-token scatter). The causal-diagonal / incomplete block, when emitted as a
   valid block index, is gathered as a partial block whose valid tokens satisfy :math:`kpos \le token\_pos`
   (inclusive, right-aligned to the query's logical position). **When ``causal = false``**, the op additionally
   enforces the upper bound :math:`kpos < \mathrm{past\_lens}[s] + \mathrm{chunk\_len}_s`, so it never reads
   unallocated or stale slots in the physical pages.
5. Applies online-softmax attention over the selected blocks.
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
blocks. **Only ``sel_indices``/``sel_count`` are indexed by ``total_q_blocks``**; the ``query``/``output`` tensors
remain per-token (``[tokens, ...]``), because a query block's selection is applied to every token inside it. The
selection for the query block covering sequence ``s``, token ``l`` is at row
``qb_start[s] + (l - subsequence_begins[s]) // B_q``, where ``qb_start`` is the running prefix sum of the
per-sequence query-block counts. When :math:`B_q > 1`, the history length of each sequence must be block-aligned:
``past_lens[s] % q_block_size == 0``.

**Paged memory organization**

The KV cache is organized into non-contiguous physical blocks of fixed ``block_size`` (the paged block size of
the main attention cache, ``PA_BLOCK_SIZE``). For sequence ``s``, the assigned physical block indices are
``block_indices[block_indices_begins[s] : block_indices_begins[s+1]]``; the logical-to-physical map is
identical to the one used by ordinary PagedAttention (``PagedAttention`` / ``PaKVReorder``), so a single
scheduler owns all paged memory. The physical KV block size ``block_size`` is **decoupled** from the producer's
selection block size ``sel_block_size``: only the divisibility constraint ``block_size % sel_block_size == 0``
is required, so one selected block maps to a *contiguous slice* of a physical block without forcing the
selection granularity to equal the paging granularity. When the producer shares the page table (as
``PagedQSAIndexer`` does), the page geometry must coincide with the physical cache: ``page_size == block_size``,
and the producer's ``summary_cache`` has the same first dimension ``num_blocks`` as ``key_cache``.

**Block granularity and the clean contract**

The producer emits block-level selection at granularity ``sel_block_size``. A selected *complete* selection
block is loaded whole (all ``sel_block_size`` tokens). The causal-diagonal / incomplete block is loaded as a
partial-block gather whose valid tokens satisfy :math:`kpos \le token\_pos` (inclusive). The plan is a
**strictly ascending, duplicate-free** sequence of block indices: the op gathers exactly the blocks named in the
contiguous prefix ``sel_indices[:sel_count]`` and never re-derives the block selection. An empty selection
(``sel_count == 0``) yields a zero output vector for that row. The kernel handles the boundary where ``sel_count``
is smaller than the selection width (short context / prefill) directly from ``sel_count``.

**Causal responsibility and the slot-read upper bound**

Causality is split across the producer and the consumer:

- **Producer (e.g. ``PagedQSAIndexer``) — block-level coarse filter (Block Eligibility).** The producer ensures
  no *future / not-yet-written complete block* is ever placed in the plan: it selects only the causally-valid
  complete blocks (block :math:`b` eligible iff :math:`(b{+}1) \cdot B_k - 1 \le pos`) plus the causal diagonal.
  The plan itself is position-free.
- **Consumer (this op) — token-level fine filter (Token-level Precision).** The consumer is the final authority
  on token-level causality: it gathers the selected blocks and imposes :math:`kpos \le token\_pos` (including
  the diagonal, self-attended). This is what guarantees bit-correct generative attention regardless of the
  producer's block granularity.

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
of blocks, per the selection contract) and block-granular selection (``sel_block_size`` may be finer than the
physical ``block_size``). When the selection covers every block (:math:`H_{sel}=1` and ``sel_count`` equals the
full block count) and ``sel_block_size == block_size``, the kernel degenerates to plain paged causal attention.
This mirrors the XAttention design, in which sparsity is a selectable execution mode of the paged-attention
primitive rather than a separate operator. The intended final form is therefore to absorb ``SparsePA`` into
``PagedAttention`` as an optional sparse-plan input group (mutually exclusive with the XAttention threshold
path); ``SparsePA`` remains a standalone internal dev op until that fusion lands.

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

    def sparse_pa(query, key, value, key_cache, value_cache, block_indices, block_indices_begins,
                  subsequence_begins, past_lens,
                  sel_indices, sel_count, qb_start,
                  *, num_heads, num_kv_heads, h_sel, k_head_size, v_head_size,
                  block_size, sel_block_size, q_block_size=1, scale, causal):
        # query:            [tokens, num_heads * k_head_size]  (projected/RMSNorm'ed/RoPE'ed; per token)
        # key, value:       [tokens, num_kv_heads * D]          (current-step tokens, in-place written)
        # key_cache:        [num_blocks, num_kv_heads, block_size, k_head_size]
        # value_cache:      [num_blocks, num_kv_heads, block_size, v_head_size]
        # block_indices:    [total_logical_blocks]   logical -> physical block map
        # block_indices_begins: [num_sequences + 1]  split pointers into block_indices
        # subsequence_begins:   [num_sequences + 1]  split pointers into the token stream
        # past_lens:        [num_sequences]  (logical KV length per sequence before this step)
        # sel_indices:      [total_q_blocks, H_sel, K_max]  (int32; strictly ascending, -1 padded past sel_count)
        # sel_count:        [total_q_blocks, H_sel]         (int32; valid block count)
        # qb_start:         [num_sequences + 1]  running prefix sum of per-sequence query-block counts
        #                   (qb_start[s] = sum_{s'<s} ceil(chunk_len_{s'} / Bq); qb_start[-1] == total_q_blocks)
        tokens = query.shape[0]
        Hq, Hkv = num_heads, num_kv_heads
        Dh, Dv = k_head_size, v_head_size
        B = block_size
        Bk = sel_block_size
        Bq = q_block_size
        blk_per_page = B // Bk                        # selection blocks per physical page
        q_per_sel = Hq // h_sel                       # query heads per selection head (Hq % h_sel == 0)

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
                row = qb_start[s] + (t - subsequence_begins[s]) // Bq      # query-block row in sel_indices
                token_pos = past_lens[s] + (t - subsequence_begins[s])     # global logical position
                for h in range(Hq):
                    sh = h // q_per_sel                   # selection head of query head h
                    kv = h // (Hq // Hkv)                 # KV head of query head h
                    nblocks = sel_count[row, sh]
                    idx = sel_indices[row, sh, :nblocks]          # valid logical block indices (ascending set)
                    # Map each selected selection-block to its physical block + in-block selection offset.
                    phys = phys_seq[idx // blk_per_page]          # physical block ids [nblocks]
                    offs = (idx % blk_per_page) * Bk              # in-block selection offset [nblocks]
                    # Gather the selected K/V tokens at sel_block_size granularity. The absolute KV position
                    # of each gathered token is derived from its logical block index: kpos = idx*Bk + off.
                    K_g, V_g = gather_physical_blocks(key_cache, value_cache,
                                                      phys, offs, Bk)   # [n_sel*Bk, Hkv, Dh], [n_sel*Bk, Hkv, Dv]
                    kpos = concat([idx[:, None] * Bk + arange(Bk)[None, :]], axis=-1).reshape(-1)  # abs pos
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

* *h_sel*

  * **Description**: Number of selection heads :math:`H_{sel}`, matching the producer. Must be ``1``,
    ``num_kv_heads``, or ``num_heads``.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``h_sel`` is ``1``, ``num_kv_heads``, or ``num_heads``; ``num_heads % h_sel == 0``.

* *k_head_size*

  * **Description**: Head dimension :math:`D_h` of the keys.
  * **Type**: ``int``
  * **Required**: *yes*

* *v_head_size*

  * **Description**: Head dimension :math:`D_v` of the values.
  * **Type**: ``int``
  * **Required**: *yes*

* *block_size*

  * **Description**: Number of tokens per physical cache block (``PA_BLOCK_SIZE``). Need not equal the
    producer's ``compress_ratio``; only ``block_size % sel_block_size == 0`` is required.
  * **Type**: ``int``
  * **Required**: *yes*

* *sel_block_size*

  * **Description**: Number of tokens per selected block, :math:`B_k`, matching the producer's
    ``compress_ratio``. Must divide ``block_size``. A selected block is a contiguous slice of a physical
    block.
  * **Type**: ``int``
  * **Required**: *yes*

* *q_block_size*

  * **Description**: Number of query tokens sharing a single selection, :math:`B_q`. Default ``1`` (decode /
    per-token selection, so ``total_q_blocks == tokens``). When :math:`B_q > 1` (e.g. FlashVSR block sparse with
    :math:`B_q = 128`), the first selection dimension is
    :math:`\mathrm{total\_q\_blocks} = \sum_s \lceil L_s / B_q \rceil`, and the history length of each sequence
    must be block-aligned: ``past_lens[s] % q_block_size == 0``.
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*

* *scale*

  * **Description**: Attention scale, ``1.0 / sqrt(k_head_size)``.
  * **Type**: ``float``
  * **Required**: *yes*

* *causal*

  * **Description**: Whether the causal-diagonal block is gathered as a partial block whose valid keys satisfy
    :math:`kpos \le token\_pos` (inclusive, self-attended). When ``false``, all written tokens of every
    selected block are attended — but the op still enforces the upper bound
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
  A 4D tensor of type *T* with shape ``[num_blocks, num_kv_heads, block_size, k_head_size]``.
  Physically paged key cache; read for scoring and written in place. **Required.**

* **4**: ``value_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, num_kv_heads, block_size, v_head_size]``.
  Physically paged value cache; read for gathering and written in place. **Required.**

* **5**: ``block_indices``
  A 1D tensor of type *T_IND* with shape ``[total_logical_blocks]``.
  Physical block indices for all sequences; ``total_logical_blocks = block_indices_begins[-1]``.
  **Required.**

* **6**: ``block_indices_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Splits ``block_indices`` among sequences. Sequence ``s`` uses blocks
  ``block_indices[block_indices_begins[s] : block_indices_begins[s+1]]``. **Required.**

* **7**: ``subsequence_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Splits the token stream among sequences. Sequence ``s`` uses tokens
  ``[subsequence_begins[s] : subsequence_begins[s + 1]]``. **Required.**

* **8**: ``past_lens``
  A 1D tensor of type *T_IND* with shape ``[num_sequences]``.
  Logical KV length of each sequence before this step. The global logical position of a token is
  ``past_lens[s] + (t - subsequence_begins[s])``; it determines the in-place write slot, the causal truncation
  bound, and the upper bound on valid logical positions for non-causal reads. **Required.**

* **9**: ``sel_indices``
  A 3D tensor of type *T_IND* with shape ``[total_q_blocks, h_sel, K_max]``.
  Per-query-block, per-selection-head selected *block* indices emitted by the upstream producer, as a
  **strictly ascending, duplicate-free** sequence in the contiguous prefix ``sel_indices[:sel_count]``. ``-1``
  padded only past ``sel_count``. When ``q_block_size == 1``, the first dimension ``total_q_blocks`` equals
  ``tokens``. **Required.**

* **10**: ``sel_count``
  A 2D tensor of type *T_IND* with shape ``[total_q_blocks, h_sel]``.
  Valid selected block count per selection head. An empty selection (``sel_count == 0``) yields a zero output
  vector for that row. **Required.**

* **11**: ``qb_start``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Running prefix sum of the per-sequence query-block counts: ``qb_start[0] == 0``,
  ``qb_start[s] == sum_{s'<s} ceil(chunk_len_{s'} / q_block_size)``, and
  ``qb_start[-1] == total_q_blocks``. Maps a (sequence, token) to the row of its query block in
  ``sel_indices``/``sel_count``/``query``/``output``. **Required.**


**Outputs**

* **0**: ``output``
  A 2D tensor of type *T* with shape ``[tokens, num_heads * v_head_size]``.
  Sparse attention output, concatenated across heads, one row per token.


**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32`` or ``int64``.
* ``num_heads % num_kv_heads == 0``.
* ``h_sel`` is ``1``, ``num_kv_heads``, or ``num_heads``; ``num_heads % h_sel == 0``.
* ``key_cache``/``value_cache`` second dimension equals ``num_kv_heads``; third dimension equals
  ``block_size``.
* ``block_size % sel_block_size == 0``; the producer's ``sel_block_size`` (``compress_ratio``) must divide the
  physical ``block_size``. When the producer shares the page table, ``page_size == block_size`` and the
  producer's summary cache has the same first dimension ``num_blocks``.
* ``block_indices`` values must be valid physical block indices ``< num_blocks``.
* ``query`` and ``output`` first dimension equals ``tokens``.
* ``sel_indices``/``sel_count`` first dimension equals ``total_q_blocks``, where
  ``total_q_blocks == qb_start[-1] == sum_s ceil(chunk_len_s / q_block_size)``; when ``q_block_size == 1`` this
  equals ``tokens``.
* ``sel_indices`` last dimension ``K_max`` is **determined by the producer**: ``block_topk + 1`` for a QSA-style
  producer, or ``ceil(S / sel_block_size)`` when the selection is produced from a block-level mask conversion.
* ``sel_count[row, sh] <= K_max`` for every row ``row`` and selection head ``sh``; the valid prefix
  ``sel_indices[row, sh, :sel_count[row, sh]]`` is **strictly ascending** and contains no ``-1`` entries.
* ``block_indices_begins[0] == 0`` and ``block_indices_begins[-1] == len(block_indices)`` (block splits);
  ``subsequence_begins[0] == 0`` and ``subsequence_begins[-1] == tokens`` (token splits).
* ``past_lens[s]`` equals the number of tokens already written for sequence ``s``; the write slot
  ``past_lens[s] + (t - subsequence_begins[s])`` must lie within an allocated physical block.
* When ``q_block_size > 1``, ``past_lens[s] % q_block_size == 0`` for every sequence ``s``.
* When ``causal`` is true, the causal-diagonal block is read only up to the query's logical position inclusive
  (:math:`kpos \le token\_pos`). The op **always** enforces the upper bound
  :math:`kpos < \mathrm{past\_lens}[s] + \mathrm{chunk\_len}_s`, so a non-causal gather never reads unallocated
  or stale slots in the physical pages.
* When a row's selection is empty (``sel_count == 0``), the op produces a zero output vector for that row.


**Types**

* *T*: any floating point type.
* *T_IND*: ``int32`` or ``int64``.


**Examples**

**Example 1 — Qwen4 Token Sparse** (:math:`B_q = 1`, :math:`B_k = 4`, :math:`H_{sel} = 1`).

A batch of two sequences, each with a fully-paged KV cache, attending with ``compress_ratio = 4`` (so
``K_max = block_topk + 1 = 3`` blocks including the causal diagonal), ``block_size = 8`` (a multiple of
``sel_block_size = 4``), ``num_heads = 24``, ``num_kv_heads = 2``, ``h_sel = 1``. With ``B_q = 1``, the first
selection dimension ``total_q_blocks == tokens``; the example shows ``tokens = 4`` (e.g. two sequences of two
tokens each, ``chunk_len = [2, 2]``, so ``qb_start = [0, 2, 4]``). ``sel_indices`` encodes, per query token, the
top-``K`` complete blocks plus the causal diagonal. When ``causal = true`` the diagonal block is gathered only up
to :math:`kpos \le token\_pos`.

.. code-block:: xml
   :force:

   <layer ... type="SparsePA">
       <data num_heads="24" num_kv_heads="2" h_sel="1" k_head_size="256" v_head_size="256"
             block_size="8" sel_block_size="4" q_block_size="1" scale="0.0625" causal="true"/>
       <input>
           <port id="0">   <!-- query: [tokens, Hq*Dh] -->
               <dim>4</dim><dim>6144</dim>
           </port>
           <port id="1">   <!-- key: [tokens, Hkv*Dh] -->
               <dim>4</dim><dim>512</dim>
           </port>
           <port id="2">   <!-- value: [tokens, Hkv*Dv] -->
               <dim>4</dim><dim>512</dim>
           </port>
           <port id="3">   <!-- key_cache: [num_blocks, Hkv, block_size, Dh] -->
               <dim>8</dim><dim>2</dim><dim>8</dim><dim>256</dim>
           </port>
           <port id="4">   <!-- value_cache: [num_blocks, Hkv, block_size, Dv] -->
               <dim>8</dim><dim>2</dim><dim>8</dim><dim>256</dim>
           </port>
           <port id="5">   <!-- block_indices: [total_logical_blocks] -->
               <dim>6</dim>
           </port>
           <port id="6">   <!-- block_indices_begins: [num_sequences+1] -->
               <dim>3</dim>
           </port>
           <port id="7">   <!-- subsequence_begins: [num_sequences+1] -->
               <dim>3</dim>
           </port>
           <port id="8">   <!-- past_lens: [num_sequences] -->
               <dim>2</dim>
           </port>
           <port id="9">   <!-- sel_indices: [total_q_blocks, H_sel, K_max] (== tokens when Bq == 1) -->
               <dim>4</dim><dim>1</dim><dim>3</dim>
           </port>
           <port id="10">  <!-- sel_count: [total_q_blocks, H_sel] -->
               <dim>4</dim><dim>1</dim>
           </port>
           <port id="11">  <!-- qb_start: [num_sequences+1] -->
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="12">  <!-- output: [tokens, Hq*Dv] -->
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
``sel_indices[0, :, :] = [[0, 2, 3, -1, ...], ...]`` (replicated across the :math:`H_{sel}` selection heads) and
``sel_count[0, :] = [3, 3, ...]``. With :math:`B_k = 128` and :math:`S = 512` there are ``S / B_k = 4`` logical
blocks; the selection covers ``3`` of them. Non-causal (:math:`causal = \mathrm{false}`), so all *written* tokens
of every selected block are attended — the op still applies the upper bound
:math:`kpos < \mathrm{past\_lens}[s] + \mathrm{chunk\_len}_s` (here ``0 + 512 = 512``), which excludes the
out-of-range tail of the final partial block and never touches unallocated/stale slots. **The input tokens must be
pre-permuted into window-major order** (FlashVSR ``WindowPartition3D``) so that each :math:`B_q`/:math:`B_k` block
is exactly one 3D window.

.. code-block:: xml
   :force:

   <layer ... type="SparsePA">
       <data num_heads="12" num_kv_heads="12" h_sel="12" k_head_size="128" v_head_size="128"
             block_size="128" sel_block_size="128" q_block_size="128" scale="0.088388" causal="false"/>
       <input>
           <port id="0">   <!-- query: [tokens, Hq*Dh] -->
               <dim>512</dim><dim>1536</dim>
           </port>
           <port id="1">   <!-- key: [tokens, Hkv*Dh] -->
               <dim>512</dim><dim>1536</dim>
           </port>
           <port id="2">   <!-- value: [tokens, Hkv*Dv] -->
               <dim>512</dim><dim>1536</dim>
           </port>
           <port id="3">   <!-- key_cache: [num_blocks, Hkv, block_size, Dh] -->
               <dim>4</dim><dim>12</dim><dim>128</dim><dim>128</dim>
           </port>
           <port id="4">   <!-- value_cache: [num_blocks, Hkv, block_size, Dv] -->
               <dim>4</dim><dim>12</dim><dim>128</dim><dim>128</dim>
           </port>
           <port id="5">   <!-- block_indices: [total_logical_blocks] -->
               <dim>4</dim>
           </port>
           <port id="6">   <!-- block_indices_begins: [num_sequences+1] -->
               <dim>2</dim>
           </port>
           <port id="7">   <!-- subsequence_begins: [num_sequences+1] -->
               <dim>2</dim>
           </port>
           <port id="8">   <!-- past_lens: [num_sequences] -->
               <dim>1</dim>
           </port>
           <port id="9">   <!-- sel_indices: [total_q_blocks, H_sel, K_max] -->
               <dim>4</dim><dim>12</dim><dim>4</dim>
           </port>
           <port id="10">  <!-- sel_count: [total_q_blocks, H_sel] -->
               <dim>4</dim><dim>12</dim>
           </port>
           <port id="11">  <!-- qb_start: [num_sequences+1] -->
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="12">  <!-- output: [tokens, Hq*Dv] -->
               <dim>512</dim><dim>1536</dim>
           </port>
       </output>
   </layer>