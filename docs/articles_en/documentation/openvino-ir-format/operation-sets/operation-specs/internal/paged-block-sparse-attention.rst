PagedBlockSparseAttention
=========================


.. meta::
  :description: Learn about PagedBlockSparseAttention - the paged variant of BlockSparseAttention
                that writes the key-value cache in paged blocks and attends to the dynamically
                selected key blocks and the query's incomplete block.

**Versioned name**: *PagedBlockSparseAttention*

**Category**: *Internal*

**Short description**: *PagedBlockSparseAttention* is the paged variant of
:doc:`BlockSparseAttention <block-sparse-attention>`. It processes the tokens of several sequences
packed into one batch, writes their keys and values into a block-paged KV cache, and computes
grouped-query attention in which every token attends only to the tokens of its selected key blocks
and of the incomplete block that contains it.

**Detailed description**:

The graph flow of a QSA layer with a paged KV cache is :doc:`PagedQSAPrepare <paged-qsa-prepare>` ->
:doc:`PagedQSAIndexer <paged-qsa-indexer>` -> *PagedBlockSparseAttention*.

The attention is the causal mode of *BlockSparseAttention*. Blocks consist of ``key_block_size``
consecutive tokens of a sequence; a token at the position ``p`` attends to:

* all tokens of the first ``selected_block_count`` blocks listed in its row of
  ``selected_block_indices`` for the selection head of the attention head, produced by
  *PagedQSAIndexer*;
* the tokens of the incomplete block that contains it,
  ``floor((p + 1) / key_block_size) * key_block_size .. p``: at most ``key_block_size - 1`` tokens,
  including the query itself, and none when ``p + 1`` is a multiple of ``key_block_size``.

``selected_block_indices`` holds ``Hs`` selections per token. Query head ``h`` uses the selection
``h // (H / Hs)``; with ``Hs = 1`` all query heads of a token share one selection, the case of QSA and
DSA. There is no other causal mask. With
``key_block_size = 4`` this is the attention of Qwen Sparse Attention (QSA, Qwen3.8-Flash-Next);
with ``key_block_size = 1`` the incomplete block is always empty and this is the sparse attention
of DeepSeek Sparse Attention (DSA, DeepSeek-V3.2, GLM-MoE-DSA), with the multi-head latent attention
in its absorbed form as described for *BlockSparseAttention*.

**Paged memory management**

The KV cache and its management are the same as for
:doc:`PagedAttentionExtension <paged-attention>`:

* The host (OpenVINO GenAI, or another scheduler) allocates the physical blocks of ``Bs`` tokens and
  passes the block table ``block_indices`` / ``block_indices_begins`` together with ``past_lens``
  and ``subsequence_begins``. ``key_cache`` and ``value_cache`` have the layout of
  *PagedAttentionExtension*, so they are allocated in the same way.
* The operation writes the keys and values of the new tokens in place, at the slots given by the
  block table, and reads the cached tokens of each sequence from it. The token at the position
  ``pos`` of the sequence ``s`` is stored in the physical block
  ``block_indices[block_indices_begins[s] + pos // Bs]`` at the offset ``pos % Bs``.
* *PagedQSAPrepare* of the same layer uses the same block table for its index caches, so ``Bs``
  must be divisible by ``key_block_size``.

``selected_block_indices`` holds logical block indices within the sequence of each token. The
operation maps every selected block to its tokens and the tokens to their slots with the block
table. Because a block never spans two physical blocks, a selected block is a contiguous run of
``key_block_size`` rows of one physical block of ``key_cache`` and ``value_cache``.

*PagedBlockSparseAttention* provides functionality according to the following pseudo-code using
``numpy``:

.. code-block:: py
   :force:

    def PagedBlockSparseAttention(query, key, value, key_cache, value_cache, past_lens, subsequence_begins,
                                  block_indices, block_indices_begins, selected_block_indices,
                                  selected_block_count, scale, *, key_block_size):
        T = query.shape[0]
        Hkv, Bs, S = key_cache.shape[1], key_cache.shape[2], key_cache.shape[3]
        Sv = value_cache.shape[3]
        H = query.shape[1] // S
        Hs = selected_block_indices.shape[1]                    # selection heads
        r = key_block_size
        output = numpy.zeros((T, H * Sv))

        def slot(s, pos):
            return block_indices[block_indices_begins[s] + pos // Bs], pos % Bs

        for s in range(len(past_lens)):
            past, t_begin, t_end = past_lens[s], subsequence_begins[s], subsequence_begins[s + 1]
            # 1. cache the keys and values of the new tokens
            for t in range(t_begin, t_end):
                blk, off = slot(s, past + t - t_begin)
                key_cache[blk, :, off] = key[t].reshape(Hkv, S)
                value_cache[blk, :, off] = value[t].reshape(Hkv, Sv)
            # 2. attend to the tokens of the selected blocks and of the incomplete block of the query
            for t in range(t_begin, t_end):
                p = past + t - t_begin                          # position of the query token
                for h in range(H):
                    g = h // (H // Hs)                          # selection head of query head h
                    blocks = selected_block_indices[t, g, :int(selected_block_count[t, g])]
                    positions = numpy.concatenate([
                        (blocks[:, None] * r + numpy.arange(r)[None, :]).reshape(-1),
                        numpy.arange((p + 1) // r * r, p + 1),  # incomplete block, < r tokens
                    ]).astype(numpy.int64)
                    if positions.size == 0:
                        continue                                # no attended token: zeros
                    kvh = h // (H // Hkv)
                    k = numpy.stack([key_cache[slot(s, pos)[0], kvh, slot(s, pos)[1]] for pos in positions])
                    v = numpy.stack([value_cache[slot(s, pos)[0], kvh, slot(s, pos)[1]] for pos in positions])
                    q = query[t, h * S:(h + 1) * S]
                    attn = Softmax((k @ q) * scale, axis=-1)    # [K]
                    output[t, h * Sv:(h + 1) * Sv] = attn @ v
        return output

Properties that follow from the definition:

* For every sequence, the result equals *BlockSparseAttention* applied to the sequence with all its
  cached and new keys and values, independently of how the tokens of the sequence were split into
  inference calls and of the physical block assignment.
* Within a call, all new keys and values are written before the attention, so a token can attend to
  earlier new tokens of its sequence in the same call. Every selected block is complete at the
  position of the token, as guaranteed by *PagedQSAIndexer*.
* The result does not depend on the order of the valid entries of a row of
  ``selected_block_indices``. The entries after the first ``selected_block_count`` ones are ignored.
* A token whose row is empty and whose incomplete block is empty produces zeros.
  *PagedQSAIndexer* never produces such a token.

The sigmoid output gate of the Qwen3.8-Flash-Next attention is applied to the output by a separate
*Multiply* and is not part of this operation.


**Attributes**

* *key_block_size*

  * **Description**: number of consecutive key tokens that form one key block of the sparse
    selection, not the physical block of the paged caches (``Bs``). It must be equal to
    *key_block_size* of the *PagedQSAIndexer* that produces ``selected_block_indices``
    and divide the block size ``Bs`` of the caches.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*


**Inputs**

.. note::

   Input ports are positional and cannot be skipped, as for *PagedAttentionExtension*.

* **0**: ``query`` - 2D tensor of type *T* and shape ``[T, H * S]``: the query heads of the new
  tokens, packed across all sequences. **Required.**

* **1**: ``key`` - 2D tensor of type *T* and shape ``[T, Hkv * S]``: the key heads of the new tokens.
  **Required.**

* **2**: ``value`` - 2D tensor of type *T* and shape ``[T, Hkv * Sv]``: the value heads of the new
  tokens. **Required.**

* **3**: ``key_cache`` - 4D tensor of type *T_cache* and shape ``[B, Hkv, Bs, S]``: the paged key
  cache, updated in place. **Required.**

* **4**: ``value_cache`` - 4D tensor of type *T_cache* and shape ``[B, Hkv, Bs, Sv]``: the paged
  value cache, updated in place. **Required.**

* **5**: ``past_lens`` - 1D tensor of type ``i32`` and shape ``[B_seq]``: the number of tokens of
  every sequence already in the cache before this call. **Required.**

* **6**: ``subsequence_begins`` - 1D tensor of type ``i32`` and shape ``[B_seq + 1]``: the new tokens
  of the sequence ``s`` are the rows ``subsequence_begins[s] .. subsequence_begins[s + 1] - 1`` of
  ``query``, ``key`` and ``value``. **Required.**

* **7**: ``block_indices`` - 1D tensor of type ``i32`` and shape ``[Nb]``: the physical block IDs of
  all sequences. **Required.**

* **8**: ``block_indices_begins`` - 1D tensor of type ``i32`` and shape ``[B_seq + 1]``: the blocks
  of the sequence ``s`` are ``block_indices[block_indices_begins[s] .. block_indices_begins[s + 1] - 1]``.
  **Required.**

* **9**: ``selected_block_indices`` - 3D tensor of type ``i32`` and shape ``[T, Hs, KB]``: for every
  new token at the position ``p`` and every selection head, the logical indices of the blocks of its
  sequence it attends to. The first ``selected_block_count`` entries are valid, unique and in
  ``[0, (p + 1) // key_block_size)``; otherwise the behavior is undefined. **Required.**

* **10**: ``selected_block_count`` - 2D tensor of type ``i32`` and shape ``[T, Hs]``: the number of
  valid entries in every row of ``selected_block_indices``. **Required.**

* **11**: ``scale`` - scalar tensor of type *T*: the attention scale factor, typically
  ``1 / sqrt(S)``. **Required.**


**Outputs**

* **0**: ``output`` - 2D tensor of type *T* and shape ``[T, H * Sv]``: the result of the sparse
  attention for every new token.

.. note::

   ``key_cache`` (input 3) and ``value_cache`` (input 4) are updated in place as a side effect.


**Types**

* *T*: any supported floating-point type.

* *T_cache*: the precision of the KV cache; ``f32``, ``f16`` or ``bf16``. It may differ from *T*.


**Dimensions**

* ``T`` - total number of new tokens in the batch.

* ``B_seq`` - number of sequences in the batch.

* ``H`` - number of query heads. ``H`` must be divisible by ``Hkv``.

* ``Hkv`` - number of key and value heads.

* ``Hs`` - number of selection heads, given by dimension 1 of ``selected_block_indices``. ``Hs``
  must divide ``H``. ``1`` for QSA and DSA.

* ``S`` - head size of the query and key.

* ``Sv`` - head size of the value.

* ``B`` - number of physical blocks.

* ``Bs`` - number of tokens per physical block, divisible by ``key_block_size``.

* ``Nb`` - total number of entries in ``block_indices``.

* ``KB`` - maximum number of selected blocks per token. For the output of
  *PagedQSAIndexer*, ``KB = token_budget / key_block_size``.


**Example**

*Qwen3.8-Flash-Next, a batch with 300 new tokens, block size 32*

.. code-block:: xml
   :force:

    <layer id="2" name="paged_sparse_attention" type="PagedBlockSparseAttention" version="extension">
        <data key_block_size="4"/>
        <input>
            <port id="0" precision="BF16"> <!-- query -->
                <dim>300</dim>   <!-- T -->
                <dim>6144</dim>  <!-- H * S = 24 * 256 -->
            </port>
            <port id="1" precision="BF16"> <!-- key -->
                <dim>300</dim>   <!-- T -->
                <dim>512</dim>   <!-- Hkv * S = 2 * 256 -->
            </port>
            <port id="2" precision="BF16"> <!-- value -->
                <dim>300</dim>   <!-- T -->
                <dim>512</dim>   <!-- Hkv * Sv = 2 * 256 -->
            </port>
            <port id="3" precision="BF16"> <!-- key_cache -->
                <dim>-1</dim>    <!-- B -->
                <dim>2</dim>     <!-- Hkv -->
                <dim>32</dim>    <!-- Bs -->
                <dim>256</dim>   <!-- S -->
            </port>
            <port id="4" precision="BF16"> <!-- value_cache -->
                <dim>-1</dim>    <!-- B -->
                <dim>2</dim>     <!-- Hkv -->
                <dim>32</dim>    <!-- Bs -->
                <dim>256</dim>   <!-- Sv -->
            </port>
            <port id="5" precision="I32"> <dim>-1</dim> </port> <!-- past_lens -->
            <port id="6" precision="I32"> <dim>-1</dim> </port> <!-- subsequence_begins -->
            <port id="7" precision="I32"> <dim>-1</dim> </port> <!-- block_indices -->
            <port id="8" precision="I32"> <dim>-1</dim> </port> <!-- block_indices_begins -->
            <port id="9" precision="I32"> <!-- selected_block_indices from PagedQSAIndexer -->
                <dim>300</dim>   <!-- T -->
                <dim>1</dim>     <!-- Hs -->
                <dim>512</dim>   <!-- KB -->
            </port>
            <port id="10" precision="I32"> <!-- selected_block_count -->
                <dim>300</dim>   <!-- T -->
                <dim>1</dim>     <!-- Hs -->
            </port>
            <port id="11" precision="BF16"/> <!-- scale -->
        </input>
        <output>
            <port id="12" precision="BF16">
                <dim>300</dim>   <!-- T -->
                <dim>6144</dim>  <!-- H * Sv -->
            </port>
        </output>
    </layer>
