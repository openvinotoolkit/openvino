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
*model-specific* producer (e.g. ``QSAIndexer``). It is the paged-memory sibling of ``SparseSDPA`` and a
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
   (inclusive, right-aligned to the query's logical position).
5. Applies online-softmax attention over the selected blocks.
6. Writes the per-head output.

**Paged memory organization**

The KV cache is organized into non-contiguous physical blocks of fixed ``block_size`` (the paged block size of
the main attention cache, ``PA_BLOCK_SIZE``). For sequence ``s``, the assigned physical block indices are
``block_indices[block_indices_begins[s] : block_indices_begins[s+1]]``; the logical-to-physical map is
identical to the one used by ordinary PagedAttention (``PagedAttention`` / ``PaKVReorder``), so a single
scheduler owns all paged memory. The physical KV block size ``block_size`` is **decoupled** from the producer's
selection block size ``sel_block_size``: only the divisibility constraint ``block_size % sel_block_size == 0``
is required, so one selected block maps to a *contiguous slice* of a physical block without forcing the
selection granularity to equal the paging granularity. When the producer shares the page table (as ``QSAIndexer``
does), the page geometry must coincide with the physical cache: ``page_size == block_size``, and the producer's
``summary_cache`` has the same first dimension ``num_blocks`` as ``key_cache``.

**Block granularity and the clean contract**

The producer emits block-level selection at granularity ``sel_block_size``. A selected *complete* selection
block is loaded whole (all ``sel_block_size`` tokens). The causal-diagonal / incomplete block is loaded as a
partial-block gather whose valid tokens satisfy :math:`kpos \le token\_pos` (inclusive). The plan is an
**unordered, duplicate-free set** of block indices: the op gathers exactly the blocks named in the contiguous
prefix ``sel_indices[:sel_count]`` and never re-derives the block selection. An empty selection (``sel_count ==
0``) yields a zero output vector for that row. The kernel handles the boundary where ``sel_count`` is smaller
than the selection width (short context / prefill) directly from ``sel_count``.

**Causal responsibility**

Causality is split across the producer and the consumer:

- **Producer (e.g. ``QSAIndexer``) — block-level coarse filter (Block Eligibility).** The producer ensures no
  *future / not-yet-written complete block* is ever placed in the plan: it selects only the causally-valid
  complete blocks (block :math:`b` eligible iff :math:`(b{+}1) \cdot B_k - 1 \le pos`) plus the causal diagonal.
  The plan itself is position-free.
- **Consumer (this op) — token-level fine filter (Token-level Precision).** The consumer is the final authority
  on token-level causality: it gathers the selected blocks and imposes :math:`kpos \le token\_pos` (including
  the diagonal, self-attended). This is what guarantees bit-correct generative attention regardless of the
  producer's block granularity.

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

Selection heads :math:`H_{sel} \in \{1, H_{kv}\}`. Query head :math:`h` maps to KV head
:math:`kv = h // kv\_groups` and to selection head :math:`sh = \min(kv, H_{sel}-1)`. When :math:`H_{sel}=1`
(QSA), all query heads share the single selection; when :math:`H_{sel}=H_{kv}`, each KV head carries its own
selection.

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def sparse_pa(query, key, value, key_cache, value_cache, block_indices, block_indices_begins,
                  subsequence_begins, past_lens,
                  sel_indices, sel_count,
                  *, num_heads, num_kv_heads, h_sel, k_head_size, v_head_size,
                  block_size, sel_block_size, scale, causal):
        # query:            [tokens, num_heads * k_head_size]   (projected/RMSNorm'ed/RoPE'ed)
        # key, value:       [tokens, num_kv_heads * D]          (current-step tokens, in-place written)
        # key_cache:        [num_blocks, num_kv_heads, block_size, k_head_size]
        # value_cache:      [num_blocks, num_kv_heads, block_size, v_head_size]
        # block_indices:    [total_logical_blocks]   logical -> physical block map
        # block_indices_begins: [num_sequences + 1]  split pointers into block_indices
        # subsequence_begins:   [num_sequences + 1]  split pointers into the token stream
        # past_lens:        [num_sequences]  (logical KV length per sequence before this step)
        # sel_indices:      [tokens, H_sel, K_max]  (int32; unordered set, -1 padded past sel_count)
        # sel_count:        [tokens, H_sel]         (int32; valid block count)
        tokens = query.shape[0]
        Hq, Hkv = num_heads, num_kv_heads
        kv_groups = Hq // Hkv
        Dh, Dv = k_head_size, v_head_size
        B = block_size
        Bk = sel_block_size
        blk_per_page = B // Bk                        # selection blocks per physical page

        q = reshape(query, (tokens, Hq, Dh))
        out = zeros((tokens, Hq, Dv))

        for s in range(num_sequences):
            for t in range(subsequence_begins[s], subsequence_begins[s + 1]):
                token_pos = past_lens[s] + (t - subsequence_begins[s])   # global logical position
                # 1. In-place KV write of the current token: physical block/slot from the shared page table.
                phys_seq = block_indices[block_indices_begins[s] : block_indices_begins[s + 1]]
                phys_block = phys_seq[token_pos // B]
                offset_in_block = token_pos % B
                key_cache[phys_block, :, offset_in_block, :] = key[t]
                value_cache[phys_block, :, offset_in_block, :] = value[t]

                for h in range(Hq):
                    kv = h // kv_groups
                    sh = min(kv, h_sel - 1)
                    nblocks = sel_count[t, sh]
                    idx = sel_indices[t, sh, :nblocks]             # valid logical block indices (set)
                    # Map each selected selection-block to its physical block + in-block selection offset.
                    phys = phys_seq[idx // blk_per_page]           # physical block ids [nblocks]
                    offs = (idx % blk_per_page) * Bk               # in-block selection offset [nblocks]
                    # Gather the selected K/V tokens at sel_block_size granularity. The absolute KV position
                    # of each gathered token is derived from its logical block index: kpos = idx*Bk + off.
                    K_g, V_g = gather_physical_blocks(key_cache, value_cache,
                                                      phys, offs, Bk)   # [n_sel*Bk, Hkv, Dh], [n_sel*Bk, Hkv, Dv]
                    kpos = concat([idx[:, None] * Bk + arange(Bk)[None, :]], axis=-1).reshape(-1)  # absolute pos
                    scores = sum(q[t, h] * K_g[:, kv, :], axis=-1) * scale   # [n_sel*Bk]
                    # Token-level fine filter: keep only keys at positions <= the query's logical position
                    # (self-inclusive). A fully-masked row yields a zero output vector (NaN guard).
                    eff = ones(kpos.shape[0], dtype=bool)
                    if causal:
                        eff = eff & (kpos <= token_pos)
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

  * **Description**: Number of selection heads :math:`H_{sel}`, matching the producer. Must be ``1`` or
    ``num_kv_heads``.
  * **Type**: ``int``
  * **Required**: *yes*

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

* *scale*

  * **Description**: Attention scale, ``1.0 / sqrt(k_head_size)``.
  * **Type**: ``float``
  * **Required**: *yes*

* *causal*

  * **Description**: Whether the causal-diagonal block is gathered as a partial block whose valid keys satisfy
    :math:`kpos \le token\_pos` (inclusive, self-attended). When ``false``, all tokens of every selected block
    are attended.
  * **Type**: ``bool``
  * **Default value**: ``true``
  * **Required**: *no*


**Inputs**

* **0**: ``query``
  A 2D tensor of type *T* with shape ``[tokens, num_heads * k_head_size]``.
  Current-step query hidden states, already projected, RMSNorm'ed and RoPE'ed by the attention layer.
  **Required.**

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
  ``past_lens[s] + (t - subsequence_begins[s])``; it determines the in-place write slot and the causal truncation
  bound. **Required.**

* **9**: ``sel_indices``
  A 3D tensor of type *T_IND* with shape ``[tokens, h_sel, K_max]``.
  Per-query, per-selection-head selected *block* indices emitted by the upstream producer, as an **unordered,
  duplicate-free set** in the contiguous prefix ``sel_indices[:sel_count]``. ``-1`` padded only past
  ``sel_count``. **Required.**

* **10**: ``sel_count``
  A 2D tensor of type *T_IND* with shape ``[tokens, h_sel]``.
  Valid selected block count per selection head. An empty selection (``sel_count == 0``) yields a zero output
  vector for that row. **Required.**


**Outputs**

* **0**: ``output``
  A 2D tensor of type *T* with shape ``[tokens, num_heads * v_head_size]``.
  Sparse attention output, concatenated across heads.


**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32`` or ``int64``.
* ``num_heads % num_kv_heads == 0``.
* ``h_sel`` is ``1`` or ``num_kv_heads``.
* ``key_cache``/``value_cache`` second dimension equals ``num_kv_heads``; third dimension equals
  ``block_size``.
* ``block_size % sel_block_size == 0``; the producer's ``sel_block_size`` (``compress_ratio``) must divide the
  physical ``block_size``. When the producer shares the page table, ``page_size == block_size`` and the
  producer's summary cache has the same first dimension ``num_blocks``.
* ``block_indices`` values must be valid physical block indices ``< num_blocks``.
* ``sel_indices`` first dimension equals ``tokens``; last dimension ``K_max`` is the producer's
  ``block_topk + 1``.
* ``sel_count[t, sh] <= K_max`` for every token ``t`` and selection head ``sh``; the valid prefix
  ``sel_indices[t, sh, :sel_count[t, sh]]`` contains no ``-1`` entries.
* ``block_indices_begins[0] == 0`` and ``block_indices_begins[-1] == len(block_indices)`` (block splits);
  ``subsequence_begins[0] == 0`` and ``subsequence_begins[-1] == tokens`` (token splits).
* ``past_lens[s]`` equals the number of tokens already written for sequence ``s``; the write slot
  ``past_lens[s] + (t - subsequence_begins[s])`` must lie within an allocated physical block.
* When ``causal`` is true, the causal-diagonal block is read only up to the query's logical position inclusive
  (:math:`kpos \le token\_pos`).
* When a row's selection is empty (``sel_count == 0``), the op produces a zero output vector for that row.


**Types**

* *T*: any floating point type.
* *T_IND*: ``int32`` or ``int64``.


**Example**

The example below shows a batch of two sequences, each with a fully-paged KV cache, attending with
``budget = 8``, ``compress_ratio = 4`` (so ``K_max = block_topk + 1 = 3`` blocks including the causal
diagonal), ``block_size = 8`` (a multiple of ``sel_block_size = 4``), ``num_heads = 24``,
``num_kv_heads = 2``, ``h_sel = 1``.

.. code-block:: xml
   :force:

   <layer ... type="SparsePA">
       <data num_heads="24" num_kv_heads="2" h_sel="1" k_head_size="256" v_head_size="256"
             block_size="8" sel_block_size="4" scale="0.0625" causal="true"/>
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
           <port id="9">   <!-- sel_indices: [tokens, H_sel, K_max] -->
               <dim>4</dim><dim>1</dim><dim>3</dim>
           </port>
           <port id="10">  <!-- sel_count: [tokens, H_sel] -->
               <dim>4</dim><dim>1</dim>
           </port>
       </input>
       <output>
           <port id="11">  <!-- output: [tokens, Hq*Dv] -->
               <dim>4</dim><dim>6144</dim>
           </port>
       </output>
   </layer>