PagedQSAPrepare
===============


.. meta::
  :description: Learn about PagedQSAPrepare - the paged variant of QSAPrepare that maintains the
                paged block index keys of Qwen Sparse Attention in place.

**Versioned name**: *PagedQSAPrepare*

**Category**: *Internal*

**Short description**: *PagedQSAPrepare* is the paged variant of :doc:`QSAPrepare <qsa-prepare>`.
It processes the raw index keys of the tokens of several sequences packed into one batch and updates
in place the paged block index keys from which :doc:`PagedSparseAttentionIndexer <paged-sparse-attention-indexer>` selects.

**Detailed description**:

The graph flow of a QSA layer with a paged KV cache is *PagedQSAPrepare* ->
:doc:`PagedSparseAttentionIndexer <paged-sparse-attention-indexer>` ->
:doc:`PagedSparseAttention <paged-sparse-attention>`. The index query is prepared by
regular operations of the graph (*RMS* and *RoPE*, as the query of the main attention) and goes to
*PagedSparseAttentionIndexer* directly; *PagedQSAPrepare* only computes the block keys.

The computation is the one of *QSAPrepare*, but the tokens of a batch belong to sequences at
different stages, and a block of a sequence may start in one inference call and be completed in a
later one. *PagedQSAPrepare* therefore keeps the state it needs in three paged caches, updated in
place:

* ``index_key_cache``: the raw index key of every token, before normalization and rotation;
* ``pa_block_index_keys``: the final key of every complete block, mean-pooled, RMS-normalized and
  rotated with the RoPE of the first token of the block. This is the cache that *PagedSparseAttentionIndexer*
  reads;
* ``index_block_rope_cache``: the RoPE ``cos`` and ``sin`` of the first token of every block, needed
  when the block is completed in a later call.

This corresponds to the key part of the ``qsa_prepare`` kernel of vLLM, which writes the raw keys
into a ring buffer and the completed block keys into a compressed key cache.

**Paged memory management**

The responsibilities are the same as for :doc:`PagedAttentionExtension <paged-attention>`:

* The host (OpenVINO GenAI, or another scheduler) allocates the physical blocks and passes the block
  table ``block_indices`` / ``block_indices_begins`` together with ``past_lens`` and
  ``subsequence_begins``. The three caches use **the same block table as the KV cache** of the
  layer: a physical block ``blk`` that holds ``Bs`` tokens of a sequence in the key and value caches
  holds their raw index keys in ``index_key_cache[blk]``, and the keys and the RoPE of the
  ``Bs / key_block_size`` blocks formed by these tokens in ``pa_block_index_keys[blk]`` and
  ``index_block_rope_cache[blk]``. The host allocates the three caches with the same number of
  physical blocks as the KV cache. ``Bs`` must be divisible by ``key_block_size``, so that a block
  never spans two physical blocks.
* The operation writes the caches: the raw index keys of the new tokens, the ``cos`` / ``sin`` of
  every new block start, and the key of every block completed by the new tokens.

The token at the position ``pos`` of the sequence ``s`` is stored in the physical block
``block_indices[block_indices_begins[s] + pos // Bs]`` at the offset ``pos % Bs``; the logical block
``b`` of the sequence is stored in the physical block that holds its first token
``b * key_block_size``, in the row ``(b * key_block_size % Bs) / key_block_size``.

Because the caches are indexed by token position and share the KV-cache blocks, prefix caching
works as for the KV cache. When the host rolls back the last tokens of a sequence (for example
rejected speculative or MTP draft tokens) by passing a smaller ``past_lens``, the keys of the blocks
that are no longer complete are recomputed from the raw keys when the blocks are completed again.

**Ordering with PagedSparseAttentionIndexer.** The output of *PagedQSAPrepare* is ``pa_block_index_keys`` itself,
the same buffer after the update. It is connected to the ``pa_block_index_keys`` input of
*PagedSparseAttentionIndexer*, so the data dependency makes the selection read the block keys of the new tokens.

*PagedQSAPrepare* provides functionality according to the following pseudo-code using ``numpy``:

.. code-block:: py
   :force:

    def PagedQSAPrepare(index_key, index_key_cache, pa_block_index_keys, index_block_rope_cache, past_lens,
                        subsequence_begins, block_indices, block_indices_begins, k_norm_weight, rope_cos, rope_sin, *,
                        key_block_size, epsilon=1e-6):
        Bs, Di = index_key_cache.shape[1], index_key_cache.shape[2]
        r = key_block_size
        Dr = rope_cos.shape[-1] if rope_cos.size else 0

        def slot(s, pos):            # physical block and offset of the token at position pos of sequence s
            return block_indices[block_indices_begins[s] + pos // Bs], pos % Bs

        for s in range(len(past_lens)):
            past, t_begin, t_end = past_lens[s], subsequence_begins[s], subsequence_begins[s + 1]
            # 1. cache the raw index keys of the new tokens, and the cos / sin of every new block start
            for t in range(t_begin, t_end):
                pos = past + t - t_begin
                blk, off = slot(s, pos)
                index_key_cache[blk, off] = index_key[t]
                if Dr and pos % r == 0:
                    index_block_rope_cache[blk, off // r] = numpy.concatenate([rope_cos[t], rope_sin[t]])
            # 2. compute the key of every block completed by the new tokens: pool, normalize, rotate
            for b in range(past // r, (past + t_end - t_begin) // r):
                k = numpy.stack([index_key_cache[slot(s, b * r + j)] for j in range(r)]).astype(numpy.float32).mean(axis=0)
                if k_norm_weight.size:
                    k = k / numpy.sqrt((k * k).mean() + epsilon) * k_norm_weight
                blk, off = slot(s, b * r)
                i = off // r                                    # row of logical block b in physical block blk
                if Dr:
                    cos, sin = index_block_rope_cache[blk, i, :Dr], index_block_rope_cache[blk, i, Dr:]
                    x = k[:Dr]
                    rotated = numpy.concatenate([-x[Dr // 2:], x[:Dr // 2]])
                    k = numpy.concatenate([x * cos + rotated * sin, k[Dr:]])
                pa_block_index_keys[blk, i] = k
        return pa_block_index_keys                              # the same tensor, updated in place

Properties that follow from the definition:

* For every sequence, ``pa_block_index_keys`` holds after the call the ``block_index_keys`` that
  *QSAPrepare* computes from all raw keys of the sequence, independently of how the tokens were split
  into inference calls and of the physical block assignment.
* The normalization (``x / sqrt(mean(x^2) + epsilon) * weight``, with ``weight = 1 + w`` for
  Qwen3.8-Flash-Next), the rotation (rotate-half on the first ``Dr`` dimensions, ``cos`` / ``sin``
  given per new token with M-RoPE already resolved) and their precision are the ones of
  *QSAPrepare*. Empty ``k_norm_weight`` or tables disable the normalization or the rotation.
* For DeepSeek Sparse Attention (``key_block_size = 1``) every block is complete as soon as its token
  is processed. The graph computes the final per-token index keys itself (LayerNorm and partial
  RoPE) and passes them as ``index_key`` with empty ``k_norm_weight``, ``rope_cos``, ``rope_sin`` and
  ``index_block_rope_cache``; the operation then only writes them into ``pa_block_index_keys``.


**Attributes**

* *key_block_size*

  * **Description**: number of consecutive key tokens that form one key block of the sparse
    selection, not the physical block of the paged caches (``Bs``). It must divide the block
    size ``Bs`` of the caches and be equal to *key_block_size* of the consuming *PagedSparseAttentionIndexer*.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *epsilon*

  * **Description**: the value added to the mean square in the RMS normalization.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: ``1e-6``
  * **Required**: *no*


**Inputs**

.. note::

   Input ports are positional and cannot be skipped. Features that are not used are disabled by
   connecting an empty tensor (zero elements), as for *PagedAttentionExtension*.

* **0**: ``index_key`` - 2D tensor of type *T* and shape ``[T, Di]``: the raw index key of every new
  token, before normalization and rotation, packed across all sequences. **Required.**

* **1**: ``index_key_cache`` - 3D tensor of type *T_cache* and shape ``[B, Bs, Di]``: the raw index
  keys of the cached tokens, updated in place. **Required.**

* **2**: ``pa_block_index_keys`` - 3D tensor of type *T_cache* and shape
  ``[B, Bs / key_block_size, Di]``: the keys of the complete blocks, updated in place. **Required.**

* **3**: ``index_block_rope_cache`` - 3D tensor of type *T_cache* and shape
  ``[B, Bs / key_block_size, 2 * Dr]``, or an empty tensor: ``rope_cos`` concatenated with
  ``rope_sin`` of the first token of every block, updated in place. **Required (empty tensor = no
  rotation).**

* **4**: ``past_lens`` - 1D tensor of type ``i32`` and shape ``[B_seq]``: the number of tokens of
  every sequence already processed in earlier calls. **Required.**

* **5**: ``subsequence_begins`` - 1D tensor of type ``i32`` and shape ``[B_seq + 1]``: the new tokens
  of the sequence ``s`` are the rows ``subsequence_begins[s] .. subsequence_begins[s + 1] - 1``.
  **Required.**

* **6**: ``block_indices`` - 1D tensor of type ``i32`` and shape ``[Nb]``: the physical block IDs of
  all sequences, the same block table as for the KV cache. **Required.**

* **7**: ``block_indices_begins`` - 1D tensor of type ``i32`` and shape ``[B_seq + 1]``: the blocks
  of the sequence ``s`` are ``block_indices[block_indices_begins[s] .. block_indices_begins[s + 1] - 1]``.
  **Required.**

* **8**: ``k_norm_weight`` - 1D tensor of type *T* and shape ``[Di]``, or an empty tensor: the scale
  of the block key normalization. **Required (empty tensor = no normalization).**

* **9**: ``rope_cos`` - 2D tensor of type *T* and shape ``[T, Dr]``, or an empty tensor: the RoPE
  ``cos`` of every new token. **Required (empty tensor = no rotation).**

* **10**: ``rope_sin`` - 2D tensor of type *T* and shape ``[T, Dr]``, or an empty tensor: the RoPE
  ``sin`` of every new token. **Required (empty tensor = no rotation).**


**Outputs**

* **0**: ``pa_block_index_keys`` - 3D tensor of type *T_cache* and shape
  ``[B, Bs / key_block_size, Di]``: input 2 after the update, the same buffer. It is connected to the
  ``pa_block_index_keys`` input of *PagedSparseAttentionIndexer*.

.. note::

   ``index_key_cache`` (input 1), ``pa_block_index_keys`` (input 2) and ``index_block_rope_cache``
   (input 3) are updated in place as a side effect. Their initial content does not matter: every row
   is written before it is read.


**Types**

* *T*: any supported floating-point type. The pooling and the normalization are computed with at
  least ``f32`` precision.

* *T_cache* - cache precision; may differ from *T*. Every cache has its own precision, independently of
  the other caches.

  * Allowed for ``index_key_cache`` (input 1): ``u4``, ``u8``, ``f16``, ``f32``, ``bf16``.
  * Allowed for ``pa_block_index_keys`` (input 2): ``u4``, ``u8``, ``f16``, ``f32``, ``bf16``.
  * Allowed for ``index_block_rope_cache`` (input 3): ``f16``, ``f32``, ``bf16``. It holds RoPE cos / sin
    values and is not quantized, as ``rotation_trig_lut`` of *PagedAttentionExtension*.


**Dimensions**

* ``T`` - total number of new tokens in the batch.

* ``B_seq`` - number of sequences in the batch.

* ``Di`` - head size of the index key.

* ``Dr`` - number of rotated dimensions, even, ``Dr <= Di``.

* ``B`` - number of physical blocks, the same as in the KV cache.

* ``Bs`` - number of tokens per physical block, the same as in the KV cache, divisible by
  ``key_block_size``.

* ``Nb`` - total number of entries in ``block_indices``.


**Example**

*Qwen3.8-Flash-Next, a batch with 300 new tokens, block size 32*

.. code-block:: xml
   :force:

    <layer id="1" name="paged_qsa_prepare" type="PagedQSAPrepare" version="extension">
        <data key_block_size="4" epsilon="1e-06"/>
        <input>
            <port id="0" precision="BF16"> <!-- index_key, raw -->
                <dim>300</dim>  <!-- T -->
                <dim>128</dim>  <!-- Di -->
            </port>
            <port id="1" precision="BF16"> <!-- index_key_cache -->
                <dim>-1</dim>   <!-- B -->
                <dim>32</dim>   <!-- Bs -->
                <dim>128</dim>  <!-- Di -->
            </port>
            <port id="2" precision="BF16"> <!-- pa_block_index_keys -->
                <dim>-1</dim>   <!-- B -->
                <dim>8</dim>    <!-- Bs / key_block_size -->
                <dim>128</dim>  <!-- Di -->
            </port>
            <port id="3" precision="BF16"> <!-- index_block_rope_cache -->
                <dim>-1</dim>   <!-- B -->
                <dim>8</dim>    <!-- Bs / key_block_size -->
                <dim>128</dim>  <!-- 2 * Dr -->
            </port>
            <port id="4" precision="I32"> <dim>-1</dim> </port> <!-- past_lens -->
            <port id="5" precision="I32"> <dim>-1</dim> </port> <!-- subsequence_begins -->
            <port id="6" precision="I32"> <dim>-1</dim> </port> <!-- block_indices -->
            <port id="7" precision="I32"> <dim>-1</dim> </port> <!-- block_indices_begins -->
            <port id="8" precision="BF16"> <dim>128</dim> </port> <!-- k_norm_weight = 1 + k_layernorm.weight -->
            <port id="9" precision="BF16"> <!-- rope_cos -->
                <dim>300</dim>  <!-- T -->
                <dim>64</dim>   <!-- Dr -->
            </port>
            <port id="10" precision="BF16"> <!-- rope_sin -->
                <dim>300</dim>  <!-- T -->
                <dim>64</dim>   <!-- Dr -->
            </port>
        </input>
        <output>
            <port id="11" precision="BF16"> <!-- pa_block_index_keys, updated: to PagedSparseAttentionIndexer -->
                <dim>-1</dim>   <!-- B -->
                <dim>8</dim>    <!-- Bs / key_block_size -->
                <dim>128</dim>  <!-- Di -->
            </port>
        </output>
    </layer>
