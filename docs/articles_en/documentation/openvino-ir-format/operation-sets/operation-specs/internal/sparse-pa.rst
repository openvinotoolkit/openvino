.. {#openvino_docs_ops_internal_SparsePA}

SparsePA
========

.. meta::
  :description: Learn about SparsePA - a unified, model-agnostic paged sparse attention operation that consumes a clean (sel_indices, sel_count, sel_block_size) selection contract and performs online-softmax attention over selected physical KV cache blocks via an explicit page table.

**Versioned name**: *SparsePA*

**Category**: *Internal*

**Short description**:
The *SparsePA* operation performs sparse, selected-block group-query attention over a *physically paged* KV
cache, using the clean ``(sel_indices, sel_count, sel_block_size)`` selection emitted by an upstream
*model-specific* producer (e.g. ``QSAIndexer``). It is the **unified, model-agnostic plan consumer** shared by
every sparse attention family (QSA, DSA, FlashVSR) and the paged-memory sibling of ``SparseSDPA``. It contains
**no producer-leaked artifacts**: no ``index_weights``, no output gate, no RoPE/norm parameters, and no
``r-1`` tail hack. It is RoPE- and norm-free; positional and normalization math lives in the main attention
layer and the upstream indexer.

**Detailed description**

*SparsePA* is the GPU-native execution form of the same sparse-attention kernel as ``SparseSDPA``. The
operation:

1. Consumes the producer's clean block selection: ``sel_indices`` (selected *block* indices per selection
   head), ``sel_count`` (valid block count per head), and ``sel_block_size`` (the block granularity).
2. Resolves each selected logical block to its physical block via ``block_indices`` /
   ``block_indices_begins``.
3. Gathers the selected K/V token states *at block granularity* (a block-gather that maps to hardware tile
   loads, rather than arbitrary per-token scatter). The causal-diagonal / incomplete block, when emitted as a
   valid block index, is gathered as a partial block bounded by ``position_ids``.
4. Applies online-softmax attention over the selected blocks.
5. Writes the per-head output.

**Paged memory organization**

The KV cache is organized into non-contiguous physical blocks of fixed ``block_size`` (the paged block size of
the main attention cache, ``PA_BLOCK_SIZE``). The producer's raw-K and summary caches are *separate* paged
tensors (see the producer spec) addressed by the same page table. For sequence ``s``, the assigned physical
block indices are ``block_indices[block_indices_begins[s] : block_indices_begins[s+1]]``; the logical-to-
physical map is identical to the one used by ordinary PagedAttention (``PagedAttention`` / ``PaKVReorder``),
so a single scheduler owns all paged memory.

**Block granularity and the clean contract**

The producer emits block-level selection at the same granularity as the physical cache block:
``sel_block_size == block_size``. A selected *complete* block is loaded whole (all ``block_size`` tokens). The
causal-diagonal / incomplete block is loaded as a partial-block gather bounded by ``position_ids``. There is no
``r-1`` tail encoding: the producer-completeness rule guarantees every block a query must attend to (sink,
sliding window, top-k scored, and the causal diagonal) is already present as a valid block index. The kernel
handles the boundary where ``sel_count`` is smaller than the selection width (short context / prefill)
directly from ``sel_count``.

**Head mapping**

Selection heads :math:`H_{sel} \in \{1, H_{kv}\}`. Query head :math:`h` maps to KV head
:math:`kv = h // kv\_groups` and to selection head :math:`sh = \min(kv, H_{sel}-1)`. When :math:`H_{sel}=1`
(QSA), all query heads share the single selection; when :math:`H_{sel}=H_{kv}`, each KV head carries its own
selection.

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def sparse_pa(query, key_cache, value_cache, block_indices, block_indices_begins,
                  sel_indices, sel_count, position_ids, subsequence_begins,
                  *, num_heads, num_kv_heads, h_sel, k_head_size, v_head_size,
                  block_size, scale, is_causal):
        # query:            [tokens, num_heads * k_head_size]   (projected/RMSNorm'ed/RoPE'ed)
        # key_cache:        [num_blocks, num_kv_heads, block_size, k_head_size]
        # value_cache:      [num_blocks, num_kv_heads, block_size, v_head_size]
        # block_indices:    [num_sequences + 1]  split pointers
        # block_indices_map:[total_logical_blocks]  logical -> physical
        # sel_indices:      [tokens, H_sel, K_max]  (int32; -1 = pad)
        # sel_count:        [tokens, H_sel]         (int32; valid block count)
        tokens = query.shape[0]
        Hq, Hkv = num_heads, num_kv_heads
        kv_groups = Hq // Hkv
        Dh, Dv = k_head_size, v_head_size
        B = block_size

        q = reshape(query, (tokens, Hq, Dh))
        out = zeros((tokens, Hq, Dv))

        for t in range(tokens):
            seq = sequence_of(t, subsequence_begins)          # owning sequence
            pos = position_ids[t]
            pblk = block_indices_map[block_indices_begins[seq] : block_indices_begins[seq+1]]
            for h in range(Hq):
                kv = h // kv_groups
                sh = min(kv, h_sel - 1)
                nblocks = sel_count[t, sh]
                idx = sel_indices[t, sh, :nblocks]            # logical block indices
                # Map logical block -> physical block; gather KV at block granularity.
                # The causal-diagonal block is partial: only tokens < pos are valid.
                phys = pblk[idx]                              # physical block ids [nblocks]
                K_g, V_g = gather_physical_blocks(key_cache, value_cache,
                                                  phys, B, pos)  # [n_tok, Hkv, Dh], [n_tok, Hkv, Dv]
                scores = sum(q[t, h] * K_g[:, kv, :], axis=-1) * scale   # [n_tok]
                probs = softmax(scores, axis=-1)              # online, over valid tokens only
                out[t, h] = sum(probs[:, None] * V_g[:, kv, :], axis=0)  # [Dv]

        return out                                           # [tokens, Hq*Dv]


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

  * **Description**: Number of tokens per physical cache block (``PA_BLOCK_SIZE``). Must equal the producer's
    ``compress_ratio`` (``sel_block_size``), so one scored block is exactly one physical cache block.
  * **Type**: ``int``
  * **Required**: *yes*

* *scale*

  * **Description**: Attention scale, ``1.0 / sqrt(k_head_size)``.
  * **Type**: ``float``
  * **Required**: *yes*

* *is_causal*

  * **Description**: Whether the causal-diagonal block is gathered as a partial block bounded by the query
    position. When ``false``, all tokens of every selected block are attended.
  * **Type**: ``bool``
  * **Default value**: ``true``
  * **Required**: *no*


**Inputs**

* **0**: ``query``
  A 2D tensor of type *T* with shape ``[tokens, num_heads * k_head_size]``.
  Current-step query hidden states, already projected, RMSNorm'ed and RoPE'ed by the attention layer.
  **Required.**

* **1**: ``key_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, num_kv_heads, block_size, k_head_size]``.
  Physically paged key cache. **Required.**

* **2**: ``value_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, num_kv_heads, block_size, v_head_size]``.
  Physically paged value cache. **Required.**

* **3**: ``block_indices``
  A 1D tensor of type *T_IND* with shape ``[total_logical_blocks]``.
  Physical block indices for all sequences; ``total_logical_blocks = block_indices_begins[-1]``.
  **Required.**

* **4**: ``block_indices_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Splits ``block_indices`` among sequences. Sequence ``s`` uses blocks
  ``block_indices[block_indices_begins[s] : block_indices_begins[s+1]]``. **Required.**

* **5**: ``sel_indices``
  A 3D tensor of type *T_IND* with shape ``[tokens, h_sel, K_max]``.
  Per-query, per-selection-head selected *block* indices emitted by the upstream producer. ``-1`` padded.
  **Required.**

* **6**: ``sel_count``
  A 2D tensor of type *T_IND* with shape ``[tokens, h_sel]``.
  Valid selected block count per selection head. **Required.**

* **7**: ``position_ids``
  A 1D tensor of type *T_IND* with shape ``[tokens]``.
  Absolute token positions; bounds the partial causal-diagonal block gather and sequence attribution.
  **Required.**

* **8**: ``subsequence_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Maps each token to its owning sequence. **Required.**


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
* ``block_size == compress_ratio`` of the upstream producer (``sel_block_size``).
* ``block_indices`` values must be valid physical block indices ``< num_blocks``.
* ``sel_indices`` first dimension equals ``tokens``; last dimension ``K_max`` is the producer's ``block_topk``.
* ``sel_count[t, sh] <= K_max`` for every token ``t`` and selection head ``sh``.
* ``subsequence_begins[0] == 0`` and ``subsequence_begins[-1] == tokens``.
* When ``is_causal`` is true, the causal-diagonal block is read only up to ``position_ids[t]``.


**Types**

* *T*: any floating point type.
* *T_IND*: ``int32`` or ``int64``.


**Example**

The example below shows a batch of two sequences, each with a fully-paged KV cache, attending with
``budget = 8``, ``compress_ratio = 4`` (so ``K_max = 2`` blocks), ``block_size = 4``, ``num_heads = 24``,
``num_kv_heads = 2``, ``h_sel = 1``.

.. code-block:: xml
   :force:

   <layer ... type="SparsePA">
       <data num_heads="24" num_kv_heads="2" h_sel="1" k_head_size="256" v_head_size="256"
             block_size="4" scale="0.0625" is_causal="true"/>
       <input>
           <port id="0">   <!-- query: [tokens, Hq*Dh] -->
               <dim>4</dim><dim>6144</dim>
           </port>
           <port id="1">   <!-- key_cache: [num_blocks, Hkv, block_size, Dh] -->
               <dim>8</dim><dim>2</dim><dim>4</dim><dim>256</dim>
           </port>
           <port id="2">   <!-- value_cache: [num_blocks, Hkv, block_size, Dv] -->
               <dim>8</dim><dim>2</dim><dim>4</dim><dim>256</dim>
           </port>
           <port id="3">   <!-- block_indices: [total_logical_blocks] -->
               <dim>6</dim>
           </port>
           <port id="4">   <!-- block_indices_begins: [num_sequences+1] -->
               <dim>3</dim>
           </port>
           <port id="5">   <!-- sel_indices: [tokens, H_sel, K_max] -->
               <dim>4</dim><dim>1</dim><dim>2</dim>
           </port>
           <port id="6">   <!-- sel_count: [tokens, H_sel] -->
               <dim>4</dim><dim>1</dim>
           </port>
           <port id="7">   <!-- position_ids: [tokens] -->
               <dim>4</dim>
           </port>
           <port id="8">   <!-- subsequence_begins: [num_sequences+1] -->
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="9">   <!-- output: [tokens, Hq*Dv] -->
               <dim>4</dim><dim>6144</dim>
           </port>
       </output>
   </layer>