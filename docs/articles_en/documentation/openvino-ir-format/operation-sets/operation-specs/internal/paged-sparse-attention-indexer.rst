PagedSparseAttentionIndexer
===========================


.. meta::
  :description: Learn about PagedSparseAttentionIndexer - the paged variant of SparseAttentionIndexer that selects the key
                blocks each token attends to from the paged block index keys.

**Versioned name**: *PagedSparseAttentionIndexer*

**Category**: *Internal*

**Short description**: *PagedSparseAttentionIndexer* is the paged variant of
:doc:`SparseAttentionIndexer <sparse-attention-indexer>`. For every token of a packed batch it scores
the complete key blocks of its sequence, read from the paged block index keys, and returns the
indices of a budget of the best-scoring blocks together with their number.

**Detailed description**:

The graph flow of a QSA layer with a paged KV cache is :doc:`PagedQSAPrepare <paged-qsa-prepare>` ->
*PagedSparseAttentionIndexer* -> :doc:`PagedSparseAttention <paged-sparse-attention>`.
*PagedSparseAttentionIndexer* only selects. The normalized and rotated index query is computed by regular
operations of the graph (*RMS* and *RoPE*, as the query of the main attention). The block keys come
from ``pa_block_index_keys``, connected to the output of *PagedQSAPrepare*: the buffer updated in place
for the new tokens. This data dependency orders the two operations. *PagedSparseAttentionIndexer* corresponds to
the ``qsa_select_paged_decode`` / ``qsa_select_paged_prefill`` kernels of vLLM.

The selection is the one of *SparseAttentionIndexer*: a token at the position ``p`` of its sequence scores the
``(p + 1) // key_block_size`` complete blocks at or before it with ``sum_h w[h] * ReLU(q[h] . k)``
over the index query heads of a selection head, and selects the ``token_budget / key_block_size``
best ones, or all of them if there are fewer. Ties are broken by the smaller block index (stable
top-k). It covers Qwen Sparse Attention (``key_block_size = 4``) and DeepSeek Sparse Attention
(``key_block_size = 1`` with ``index_weights``); see *SparseAttentionIndexer* for the model mapping and the
selection heads.

``pa_block_index_keys`` uses the block table of the KV cache: the key of the logical block ``b`` of
the sequence ``s`` is in the physical block that holds the token ``b * key_block_size``, at the row
``(b * key_block_size % Bs) / key_block_size``, with ``Bs = key_block_size * pa_block_index_keys.shape[1]``.

*PagedSparseAttentionIndexer* provides functionality according to the following pseudo-code using ``numpy``:

.. code-block:: py
   :force:

    def PagedSparseAttentionIndexer(index_query, pa_block_index_keys, past_lens, subsequence_begins, block_indices,
                        block_indices_begins, index_weights, *, key_block_size, token_budget, num_selection_heads=1):
        T = index_query.shape[0]
        rows_per_block, Di = pa_block_index_keys.shape[1], pa_block_index_keys.shape[2]
        Hi = index_query.shape[1]
        r = key_block_size
        Bs = rows_per_block * r                                 # tokens per physical block
        Hs = num_selection_heads
        G = Hi // Hs
        block_topk = token_budget // r

        def row(s, b):               # physical block and row of the block key of logical block b of sequence s
            pos = b * r
            return block_indices[block_indices_begins[s] + pos // Bs], (pos % Bs) // r

        selected_block_indices = numpy.full((T, Hs, block_topk), -1, dtype=numpy.int32)
        selected_block_count = numpy.zeros((T, Hs), dtype=numpy.int32)
        for s in range(len(past_lens)):
            past, t_begin, t_end = past_lens[s], subsequence_begins[s], subsequence_begins[s + 1]
            for t in range(t_begin, t_end):
                p = past + t - t_begin                          # position of the query token
                num_visible_blocks = (p + 1) // r
                if num_visible_blocks == 0:
                    continue
                k = numpy.stack([pa_block_index_keys[row(s, b)] for b in range(num_visible_blocks)]).astype(numpy.float32)
                q_all = index_query[t].astype(numpy.float32)              # [Hi, Di]
                w_all = index_weights[t].astype(numpy.float32) if index_weights.size else numpy.ones(Hi)
                for g in range(Hs):
                    q, w = q_all[g * G:(g + 1) * G], w_all[g * G:(g + 1) * G]
                    scores = (w[:, None] * numpy.maximum(q @ k.T, 0.0)).sum(axis=0)
                    num_selected_blocks = min(block_topk, num_visible_blocks)
                    # stable top-k: descending score, equal scores -> smaller block index first
                    blocks = numpy.argsort(-scores, kind="stable")[:num_selected_blocks]
                    selected_block_indices[t, g, :num_selected_blocks] = blocks
                    selected_block_count[t, g] = num_selected_blocks
        return selected_block_indices, selected_block_count

Properties that follow from the definition:

* For every sequence, the result equals *SparseAttentionIndexer* applied to the sequence with
  ``key_length = past_lens[s] + new_len_s`` and the ``block_index_keys`` of *QSAPrepare*,
  independently of how the tokens of the sequence were split into inference calls and of the
  physical block assignment.
* ``selected_block_indices`` holds logical block indices of the sequence of the token, not physical
  block IDs. *PagedSparseAttention* maps them to the physical blocks with the same block table.
* Within a call, a block completed by a later token of the batch is never visible to an earlier
  token: a token at the position ``p`` sees only the blocks that end at or before ``p``.
* ``selected_block_count`` is ``min(token_budget / key_block_size, (p + 1) // key_block_size)`` for
  every selection head, and the valid entries come first in every row.
* The operation reads ``pa_block_index_keys`` and writes no cache.


**Attributes**

* *key_block_size*

  * **Description**: number of consecutive key tokens that form one key block of the sparse
    selection, not the physical block of the paged caches (``Bs``). It must be equal to
    *key_block_size* of *PagedQSAPrepare* and *PagedSparseAttention*.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *token_budget*

  * **Description**: maximum number of tokens selected from complete blocks for one token.
    ``token_budget / key_block_size`` blocks are selected.
  * **Range of values**: a positive integer divisible by *key_block_size*
  * **Type**: ``int``
  * **Required**: *yes*

* *num_selection_heads*

  * **Description**: the number of selections per token, ``Hs``. The ``Hi`` index query heads are
    split into ``Hs`` consecutive groups, the scores are summed within each group, and every group
    selects its own blocks. ``1`` for QSA and DSA.
  * **Range of values**: a positive integer that divides ``Hi`` and the number of attention query
    heads of *PagedSparseAttention*
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*


**Inputs**

.. note::

   Input ports are positional and cannot be skipped. Features that are not used are disabled by
   connecting an empty tensor (zero elements), as for *PagedAttentionExtension*.

* **0**: ``index_query`` - 3D tensor of type *T* and shape ``[T, Hi, Di]``: the normalized and
  rotated index query heads of every new token, packed across all sequences. **Required.**

* **1**: ``pa_block_index_keys`` - 3D tensor of type *T_cache* and shape
  ``[B, Bs / key_block_size, Di]``: the keys of the complete blocks, the output of
  *PagedQSAPrepare*. **Required.**

* **2**: ``past_lens`` - 1D tensor of type ``i32`` and shape ``[B_seq]``: the number of tokens of
  every sequence already processed in earlier calls. **Required.**

* **3**: ``subsequence_begins`` - 1D tensor of type ``i32`` and shape ``[B_seq + 1]``: the new tokens
  of the sequence ``s`` are the rows ``subsequence_begins[s] .. subsequence_begins[s + 1] - 1``.
  **Required.**

* **4**: ``block_indices`` - 1D tensor of type ``i32`` and shape ``[Nb]``: the physical block IDs of
  all sequences, the same block table as for the KV cache. **Required.**

* **5**: ``block_indices_begins`` - 1D tensor of type ``i32`` and shape ``[B_seq + 1]``: the blocks
  of the sequence ``s`` are ``block_indices[block_indices_begins[s] .. block_indices_begins[s + 1] - 1]``.
  **Required.**

* **6**: ``index_weights`` - 2D tensor of type *T* and shape ``[T, Hi]``, or an empty tensor: the
  weight of every index query head for every new token. Empty means all weights are ``1``, as in
  Qwen3.8-Flash-Next; DSA models provide them. **Required (empty tensor = all weights 1).**


**Outputs**

* **0**: ``selected_block_indices`` - 3D tensor of type ``i32`` and shape
  ``[T, Hs, token_budget / key_block_size]``: the logical indices of the blocks selected by every
  selection head for every new token within its sequence, padded with ``-1``.

* **1**: ``selected_block_count`` - 2D tensor of type ``i32`` and shape ``[T, Hs]``: the number of
  valid entries in every row of ``selected_block_indices``.


**Types**

* *T*: any supported floating-point type. The scores are accumulated with at least ``f32``
  precision.

* *T_cache*: the precision of ``pa_block_index_keys``; ``f32``, ``f16`` or ``bf16``.


**Dimensions**

* ``T`` - total number of new tokens in the batch.

* ``B_seq`` - number of sequences in the batch.

* ``Hi`` - number of index query heads. The indexer has one shared key head.

* ``Hs`` - number of selection heads, *num_selection_heads*.

* ``Di`` - head size of the index query and key.

* ``B`` - number of physical blocks, the same as in the KV cache.

* ``Bs`` - number of tokens per physical block, divisible by ``key_block_size``.

* ``Nb`` - total number of entries in ``block_indices``.


**Example**

*Qwen3.8-Flash-Next, a batch with 300 new tokens, block size 32*

.. code-block:: xml
   :force:

    <layer id="2" name="paged_sparse_attention_indexer" type="PagedSparseAttentionIndexer" version="extension">
        <data key_block_size="4" token_budget="2048" num_selection_heads="1"/>
        <input>
            <port id="0" precision="BF16"> <!-- index_query: RMS + RoPE in the graph -->
                <dim>300</dim>  <!-- T -->
                <dim>4</dim>    <!-- Hi -->
                <dim>128</dim>  <!-- Di -->
            </port>
            <port id="1" precision="BF16"> <!-- pa_block_index_keys: output of PagedQSAPrepare -->
                <dim>-1</dim>   <!-- B -->
                <dim>8</dim>    <!-- Bs / key_block_size -->
                <dim>128</dim>  <!-- Di -->
            </port>
            <port id="2" precision="I32"> <dim>-1</dim> </port> <!-- past_lens -->
            <port id="3" precision="I32"> <dim>-1</dim> </port> <!-- subsequence_begins -->
            <port id="4" precision="I32"> <dim>-1</dim> </port> <!-- block_indices -->
            <port id="5" precision="I32"> <dim>-1</dim> </port> <!-- block_indices_begins -->
            <port id="6" precision="BF16"> <dim>0</dim> </port> <!-- index_weights: empty -->
        </input>
        <output>
            <port id="7" precision="I32"> <!-- selected_block_indices -->
                <dim>300</dim>  <!-- T -->
                <dim>1</dim>    <!-- Hs -->
                <dim>512</dim>  <!-- token_budget / key_block_size -->
            </port>
            <port id="8" precision="I32"> <!-- selected_block_count -->
                <dim>300</dim>  <!-- T -->
                <dim>1</dim>    <!-- Hs -->
            </port>
        </output>
    </layer>
