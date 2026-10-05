.. {#openvino_docs_ops_internal_SparseSDPA}

SparseSDPA
==========

.. meta::
  :description: Learn about SparseSDPA - a unified, model-agnostic contiguous-tensor sparse attention operation that consumes a clean (sel_indices, sel_count, sel_block_size) selection contract and performs online-softmax attention over the selected KV blocks without a physical page table.

**Versioned name**: *SparseSDPA*

**Category**: *Internal*

**Short description**:
The *SparseSDPA* operation performs sparse, selected-block multi-head attention over *contiguous,
KVCache-managed* K/V tensors. It is the **unified, model-agnostic plan consumer** shared by every sparse
attention family (QSA, DSA, FlashVSR). It consumes a strictly clean
``(sel_indices, sel_count, sel_block_size)`` contract from a *model-specific* producer (e.g. ``QSAIndexer``),
gathers the KV at the selected blocks, and applies online-softmax attention with GQA head mapping. It contains
**no producer-leaked artifacts**: no ``index_weights``, no output gate, no RoPE/norm parameters, and no
``r-1`` tail hack. It is intentionally RoPE- and norm-free: all positional and normalization math is owned by
the main attention layer and the upstream indexer.

**Detailed description**

*SparseSDPA* decouples *attention execution* from *sparse plan production*. Its inputs are:

- the current query (already projected by the main Q-projection, RMSNorm'ed and RoPE'ed by the attention
  layer),
- dense, history-complete K and V cache tensors produced by a standard ``KVCache`` node, and
- the clean selection produced by an upstream model-specific indexer: per-query, per-selection-head block
  indices ``sel_indices`` and their valid counts ``sel_count``, plus the block granularity
  ``sel_block_size``.

The operation:

1. **Gathers** the selected KV blocks for each query (a block-gather, not a masked matmul over the full
   history — the key bandwidth/FLOP win of sparse attention). The causal-diagonal / incomplete block, when
   emitted by the producer as a valid block index, is gathered as a partial block bounded by ``position_ids``.
2. **Scores** each selected key against the query with the attention scale ``1/sqrt(Dh)``.
3. **Online-softmax** normalizes scores over the selected set.
4. **Weighted-averages** the gathered values and concatenates across heads.

Because the selection is index-based (a fixed-width block buffer per query, see the producer's outputs), there
is no need to materialize a ``[tokens, kv_length]`` boolean mask — the exact source of the bandwidth saving.
The op supports group-query attention (GQA) with ``num_kv_heads <= num_heads`` and a clean mapping from query
heads to selection heads.

**The clean selection contract**

The producer emits block-level selection:

- ``sel_indices`` ``[T, H_sel, K_max]``: selected *block* indices per selection head, ``-1`` padded.
- ``sel_count`` ``[T, H_sel]``: valid block count per selection head.
- ``sel_block_size`` (attribute): the number of tokens per block; the consumer gathers full blocks of this size
  (and a partial prefix of the causal-diagonal block, bounded by ``position_ids``).

Producer-completeness: every block a query must attend to (sink, sliding window, top-k scored, and the causal
diagonal / incomplete block) is already present as a valid block index. The consumer never re-derives the
selection and carries no ``index_weights``, no output gate, and no ``r-1`` tail encoding.

**Head mapping**

Selection heads :math:`H_{sel} \in \{1, H_{kv}\}`. Query head :math:`h` maps to KV head
:math:`kv = h // kv\_groups` and to selection head :math:`sh = \min(kv, H_{sel}-1)`. When :math:`H_{sel}=1`
(QSA), all query heads share the single selection; when :math:`H_{sel}=H_{kv}`, each KV head carries its own
selection. Per-query-head selection for GQA is rejected by construction of the producer contract.

**Decode incremental contract**

The op is *stateless with respect to the selection*: it recomputes attention over the selected set each step.
The incrementality burden is entirely on the upstream producer (which maintains the block-summary cache). For a
decode step with :math:`B` selected blocks per query and :math:`H_{kv}` KV heads, the per-query work is
:math:`O(B \cdot sel\_block\_size \cdot D_v)` for the value gather and :math:`O(B \cdot sel\_block\_size \cdot
D_h)` for scoring — independent of total history length :math:`S`. This is what makes long-context decode
tractable: the op's cost is bounded by the budget, not the sequence length.

**Relationship to ``SparsePA``**

``SparseSDPA`` and ``SparsePA`` share the exact same mathematical kernel (score :math:`\to` online-softmax
:math:`\to` weighted-gather) and the same clean selection contract. The only difference is the KV memory
contract:

- ``SparseSDPA``: dense / KVCache-managed contiguous tensors, one physical page per sequence, identity block
  table. Best for CPU reference, early prototyping, and pipelines that already use a contiguous KVCache.
- ``SparsePA``: physically paged ``[num_blocks, Hkv, block_size, D]`` cache with an explicit
  ``block_indices`` / ``block_indices_begins`` table. Best for GPU where block granularity maps to hardware
  tiles and where paged memory management (prefix caching, continuous batching) is required.

A lowering pass may convert ``SparseSDPA`` into ``SparsePA`` (or vice versa) by synthesizing a degenerate /
real page table; the fused GPU primitive dispatch is shared.

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def sparse_sdpa(query, key_cache_dense, value_cache_dense,
                    sel_indices, sel_count, position_ids, subsequence_begins,
                    *, num_heads, num_kv_heads, h_sel, k_head_size, v_head_size,
                    sel_block_size, scale, is_causal):
        # query:             [tokens, num_heads * k_head_size]  (already projected/RMSNorm'ed/RoPE'ed)
        # key_cache_dense:   [num_seqs, num_kv_heads, S, k_head_size]  (KVCache "present" output)
        # value_cache_dense: [num_seqs, num_kv_heads, S, v_head_size]
        # sel_indices:       [tokens, H_sel, K_max]   (int32; -1 = pad)
        # sel_count:         [tokens, H_sel]          (int32; valid block count)
        tokens = query.shape[0]
        Hq, Hkv = num_heads, num_kv_heads
        kv_groups = Hq // Hkv
        Dh, Dv = k_head_size, v_head_size
        B = sel_block_size

        q = reshape(query, (tokens, Hq, Dh))                 # [tokens, Hq, Dh]
        out = zeros((tokens, Hq, Dv))

        for t in range(tokens):
            seq = sequence_of(t, subsequence_begins)         # owning sequence
            pos = position_ids[t]
            for h in range(Hq):
                kv = h // kv_groups
                sh = min(kv, h_sel - 1)
                nblocks = sel_count[t, sh]                   # number of valid blocks
                idx = sel_indices[t, sh, :nblocks]           # [nblocks] logical block indices
                # Gather KV at block granularity. The final (causal-diagonal) block is partial:
                # valid tokens in it are bounded by pos (only tokens < pos are attended).
                K_g, V_g = gather_blocks(key_cache_dense, value_cache_dense,
                                         seq, idx, B, pos)   # [n_tok, Hkv, Dh], [n_tok, Hkv, Dv]
                scores = sum(q[t, h] * K_g[:, kv, :], axis=-1) * scale   # [n_tok]
                probs = softmax(scores, axis=-1)             # online, over valid tokens only
                out[t, h] = sum(probs[:, None] * V_g[:, kv, :], axis=0)  # [Dv]

        return out                                          # [tokens, Hq*Dv]


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

* *sel_block_size*

  * **Description**: Number of tokens per selected block, matching the producer's ``compress_ratio``. The
    consumer gathers full blocks of this size; the causal-diagonal block may be partial, bounded by
    ``position_ids``.
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

* **1**: ``key_cache_dense``
  A 4D tensor of type *T* with shape ``[num_seqs, num_kv_heads, S, k_head_size]``.
  Dense key cache containing the full history (the ``KVCache`` "present" output). **Required.**

* **2**: ``value_cache_dense``
  A 4D tensor of type *T* with shape ``[num_seqs, num_kv_heads, S, v_head_size]``.
  Dense value cache containing the full history. **Required.**

* **3**: ``sel_indices``
  A 3D tensor of type *T_IND* with shape ``[tokens, h_sel, K_max]``.
  Per-query, per-selection-head selected *block* indices emitted by the upstream producer. ``-1`` padded.
  **Required.**

* **4**: ``sel_count``
  A 2D tensor of type *T_IND* with shape ``[tokens, h_sel]``.
  Valid selected block count per selection head. **Required.**

* **5**: ``position_ids``
  A 1D tensor of type *T_IND* with shape ``[tokens]``.
  Absolute token positions; bounds the partial causal-diagonal block gather. **Required.**

* **6**: ``subsequence_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Maps each token to its owning sequence (for indexing into ``key_cache_dense`` / ``value_cache_dense``).
  **Required.**


**Outputs**

* **0**: ``output``
  A 2D tensor of type *T* with shape ``[tokens, num_heads * v_head_size]``.
  The sparse attention output, concatenated across heads.


**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32`` or ``int64``.
* ``num_heads % num_kv_heads == 0``; the group size ``num_heads // num_kv_heads`` must be positive.
* ``h_sel`` is ``1`` or ``num_kv_heads``.
* ``key_cache_dense``/``value_cache_dense`` second dimension equals ``num_kv_heads``.
* ``sel_indices`` first dimension equals ``tokens``; last dimension ``K_max`` is the producer's ``block_topk``.
* ``subsequence_begins[0] == 0`` and ``subsequence_begins[-1] == tokens``; each token must map to a valid
  sequence index so the gather can index the dense caches.
* When ``is_causal`` is true, the causal-diagonal block is read only up to ``position_ids[t]``.


**Types**

* *T*: any floating point type.
* *T_IND*: ``int32`` or ``int64``.


**Example**

The example below shows a query token at position ``9`` attending over the selection produced by a
``QSAIndexer`` (see the ``QSAIndexer`` example): ``sel_indices = [[[1, 0, 2]]]``, ``sel_count = [[3]]``,
``sel_block_size = 4``, ``num_heads = 24``, ``num_kv_heads = 2``, ``h_sel = 1``. The consumer gathers blocks
``{1, 0, 2}``; block ``2`` is the causal-diagonal partial block (tokens ``8`` only, since ``pos = 9``).

.. code-block:: xml
   :force:

   <layer ... type="SparseSDPA">
       <data num_heads="24" num_kv_heads="2" h_sel="1" k_head_size="256" v_head_size="256"
             sel_block_size="4" scale="0.0625" is_causal="true"/>
       <input>
           <port id="0">   <!-- query: [tokens, Hq*Dh] -->
               <dim>2</dim><dim>6144</dim>
           </port>
           <port id="1">   <!-- key_cache_dense: [num_seqs, Hkv, S, Dh] -->
               <dim>2</dim><dim>2</dim><dim>10</dim><dim>256</dim>
           </port>
           <port id="2">   <!-- value_cache_dense: [num_seqs, Hkv, S, Dv] -->
               <dim>2</dim><dim>2</dim><dim>10</dim><dim>256</dim>
           </port>
           <port id="3">   <!-- sel_indices: [tokens, H_sel, K_max] -->
               <dim>2</dim><dim>1</dim><dim>2</dim>
           </port>
           <port id="4">   <!-- sel_count: [tokens, H_sel] -->
               <dim>2</dim><dim>1</dim>
           </port>
           <port id="5">   <!-- position_ids: [tokens] -->
               <dim>2</dim>
           </port>
           <port id="6">   <!-- subsequence_begins: [num_sequences+1] -->
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="7">   <!-- output: [tokens, Hq*Dv] -->
               <dim>2</dim><dim>6144</dim>
           </port>
       </output>
   </layer>