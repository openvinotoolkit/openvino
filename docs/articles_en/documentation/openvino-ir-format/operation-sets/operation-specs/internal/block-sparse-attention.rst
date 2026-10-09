BlockSparseAttention
====================


.. meta::
  :description: Learn about BlockSparseAttention - grouped-query scaled dot-product attention
                restricted to dynamically selected key blocks and the query's incomplete block.

**Versioned name**: *BlockSparseAttention*

**Category**: *Sequence processing*

**Short description**: *BlockSparseAttention* computes grouped-query scaled dot-product attention
in which every query token attends only to the tokens of its selected key blocks and to the tokens
of the incomplete block that contains it.

**Detailed description**:

*BlockSparseAttention* is the attention stage of dynamic, query-dependent sparse attention. The
blocks are not a fixed sparsity pattern: they are selected per query token and passed in through
``block_indices``, normally produced by :doc:`SparseAttentionIndexer <sparse-attention-indexer>`.
It covers:

* Qwen Sparse Attention (QSA) of the attention layers of Qwen3.8-Flash-Next (``qwen4_exp`` in
  Hugging Face Transformers), with blocks of ``compress_ratio = 4`` tokens. Most of this document
  uses it as the example;
* DeepSeek Sparse Attention (DSA) of DeepSeek-V3.2 and GLM-MoE-DSA, with single-token blocks
  (``compress_ratio = 1``). See the note below.

The decoupling from the selection follows inference engines:

* vLLM (``vllm/models/qwen4_exp``): the sparse paged attention kernel attends to the tokens of the
  blocks selected by its ``QSAIndexer`` class plus the incomplete block of the query, read from the
  main key/value cache.
* llama.cpp (``src/models/qwen4exp.cpp``): ``build_attn_qsa`` starts from a mask filled with
  ``-inf``, writes ``0`` at the selected positions with ``ggml_set_rows``, and runs regular
  multi-head attention with that mask.

Blocks consist of ``compress_ratio`` consecutive key tokens: block ``b`` holds the positions
``b * compress_ratio .. b * compress_ratio + compress_ratio - 1``. A query token at the position
``p`` attends to:

* all tokens of the first ``block_count`` blocks listed in its row of ``block_indices`` for the
  selection head of the attention head;
* the tokens of the incomplete block that contains it, ``floor((p + 1) / compress_ratio) * compress_ratio .. p``:
  at most ``compress_ratio - 1`` tokens, including the query itself. This part is empty when
  ``p + 1`` is a multiple of ``compress_ratio``, because the block of the query is then complete
  and attended only if it is listed in ``block_indices``.

**Selection heads.** ``block_indices`` and ``block_count`` hold ``Hs`` selections per query token.
The ``H`` attention query heads are split into ``Hs`` consecutive groups of ``H / Hs`` heads, and
query head ``h`` uses the selection ``h // (H / Hs)``. With ``Hs = 1`` all query heads of a token
share one selection, the case of QSA and DSA; with ``Hs = Hkv`` every key-value head and its group
of query heads has its own selection, as in Native Sparse Attention (NSA); with ``Hs = H`` every
query head has its own selection. The incomplete block of the query is attended by every head.

There is no other causal mask: the causality
comes from the selection. This document covers the *ScaledDotProductAttention* case: the key and
value tensors hold all tokens of the sequences contiguously, as with a stateful KV cache, rather
than in a paged cache.

*BlockSparseAttention* provides functionality according to the following pseudo-code using
``numpy``:

.. code-block:: py
   :force:

    def BlockSparseAttention(query, key, value, block_indices, block_count, scale=None, *, compress_ratio):
        N, H, L, E = query.shape
        Hkv, S = key.shape[1], key.shape[2]
        Hs = block_indices.shape[1]                             # selection heads
        r = compress_ratio
        if scale is None:
            scale = 1.0 / sqrt(E)

        mask = numpy.zeros((N, Hs, L, S), dtype=bool)
        for n in range(N):
            for g in range(Hs):
                for l in range(L):
                    p = S - L + l                               # position of the query token
                    blocks = block_indices[n, g, l, :int(block_count[n, g, l])]  # the rest is ignored
                    selected = (blocks[:, None] * r + numpy.arange(r)[None, :]).reshape(-1)
                    tail = numpy.arange((p + 1) // r * r, p + 1)    # incomplete block, < r tokens
                    mask[n, g, l, selected] = True
                    mask[n, g, l, tail] = True
        mask = numpy.repeat(mask, H // Hs, axis=1)              # selection g -> query heads of group g

        key = numpy.repeat(key, H // Hkv, axis=1)
        value = numpy.repeat(value, H // Hkv, axis=1)
        attn = (query @ numpy.swapaxes(key, -1, -2)) * scale        # [N, H, L, S]
        attn = numpy.where(mask, attn, -inf)
        attn = Softmax(attn, axis=-1)
        output = attn @ value                                       # [N, H, L, Ev]
        # a query head without selected positions produces zeros
        return numpy.where(mask.any(axis=-1)[..., None], output, 0.0)

Properties that follow from the definition:

* The ``L`` query tokens are the last ``L`` of the ``S`` key tokens: query ``l`` has the position
  ``S - L + l``.
* With the valid block indices of a row in ``[0, (p + 1) // compress_ratio)``, as produced by
  *SparseAttentionIndexer*, every attended token is at or before ``p``. A query attends to
  ``block_count * compress_ratio + (p + 1) % compress_ratio`` tokens, so the number of key and value
  rows of every query is known before any of them is read.
* ``block_count`` is the loop bound over the selected blocks of a row: an implementation reads only
  the first ``block_count`` entries and never scans the row for ``-1``. It can also size the work of
  a query and skip rows without selected blocks without reading ``block_indices``, like the valid-entry
  count column that the vLLM sparse attention kernel uses as its tile-loop bound.
* The result does not depend on the order of the valid block indices in a row or on the entries
  after the first ``block_count`` ones, which are ignored.
* With the rows produced by *SparseAttentionIndexer*, the result is identical to causal
  *ScaledDotProductAttention* as long as every complete block is selected, that is for the first
  ``token_budget + compress_ratio - 1`` positions.
* A query head attends to nothing only if ``p + 1`` is a multiple of ``compress_ratio`` and the
  ``block_count`` of its selection is ``0``. *SparseAttentionIndexer* never produces such a row.
* With ``Hs = 1`` every query head of a token attends to the same key and value rows, so an
  implementation can read every selected block once for all heads, as for multi-query attention.
  With ``Hs = Hkv`` the rows are shared within every group of query heads of one key-value head.
* The operation can be expressed with other operations by expanding ``block_indices`` and the
  incomplete block into a boolean mask and passing it to *ScaledDotProductAttention* as
  ``attention_mask``, as llama.cpp does. That costs time and memory proportional to ``S`` for every
  query, while a dedicated implementation reads only the selected blocks, each as a contiguous run
  of ``compress_ratio`` key and value rows.

The sigmoid output gate of the Qwen3.8-Flash-Next attention is applied to the output by a separate
*Multiply* and is not part of this operation.

.. note::

   With ``compress_ratio = 1``, every block is a single token and the incomplete block is always
   empty, so a query attends to exactly the tokens listed in ``block_indices``. Together with
   :doc:`SparseAttentionIndexer <sparse-attention-indexer>` configured as described there, this covers the sparse attention
   of DeepSeek Sparse Attention (DSA) models such as DeepSeek-V3.2 and GLM-MoE-DSA. Their
   multi-head latent attention maps to this operation in its absorbed form, as multi-query
   attention with ``Hkv = 1``, ``E = kv_lora_rank + qk_rope_head_dim`` (``576``) and
   ``Ev = kv_lora_rank`` (``512``), with the attention scale passed through ``scale``.


**Attributes**

* *compress_ratio*

  * **Description**: number of consecutive key tokens that form one block. It must be equal to
    *compress_ratio* of the *SparseAttentionIndexer* that produces ``block_indices``.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*


**Inputs**

* **1**: ``query`` - 4D tensor of type *T* and shape ``[N, H, L, E]``: the attention query heads.
  **Required.**

* **2**: ``key`` - 4D tensor of type *T* and shape ``[N, Hkv, S, E]``: the attention key heads of
  all tokens, including the past ones. **Required.**

* **3**: ``value`` - 4D tensor of type *T* and shape ``[N, Hkv, S, Ev]``: the attention value heads
  of all tokens, including the past ones. **Required.**

* **4**: ``block_indices`` - 4D tensor of type *T_IND* and shape ``[N, Hs, L, KB]``: for every
  selection head and every query token at the position ``p``, the indices of the key blocks it
  attends to in its first ``block_count`` entries. These entries are in
  ``[0, (p + 1) // compress_ratio)`` and unique within a row; otherwise the behavior is undefined.
  The remaining entries are ignored; *SparseAttentionIndexer* sets them to ``-1``. **Required.**

* **5**: ``block_count`` - 3D tensor of type *T_IND* and shape ``[N, Hs, L]``: the number of valid
  entries in every row of ``block_indices``, ``0 <= block_count <= KB``. Normally the
  ``block_count`` output of *SparseAttentionIndexer*. **Required.**

* **6**: ``scale`` - a scalar or single element 1D tensor of type *T*: the attention scale factor,
  used instead of the default ``1 / sqrt(E)``. **Optional.**


**Outputs**

* **1**: ``output`` - 4D tensor of type *T* and shape ``[N, H, L, Ev]``: the result of the sparse
  attention.


**Types**

* *T*: any supported floating-point type.

* *T_IND*: ``int32`` or ``int64``.


**Dimensions**

* ``N`` - batch size. The sequences of a batch are aligned to the right: query ``l`` of every
  sequence is at position ``S - L + l``. Batches with padding are not supported.

* ``H`` - number of attention query heads. ``H`` must be divisible by ``Hkv``.

* ``Hkv`` - number of attention key and value heads.

* ``Hs`` - number of selection heads, given by dimension 1 of ``block_indices`` and ``block_count``.
  ``Hs`` must divide ``H``. ``1`` for QSA and DSA.

* ``L`` - number of query tokens, ``L <= S``.

* ``S`` - number of key and value tokens, including the query tokens.

* ``E`` - head size of the attention query and key.

* ``Ev`` - head size of the attention value.

* ``KB`` - maximum number of selected blocks per query. For the output of *SparseAttentionIndexer*,
  ``KB = token_budget / compress_ratio``.


**Examples**

*Example 1: Prefill with the Qwen3.8-Flash-Next configuration*

.. code-block:: xml
   :force:

    <layer id="3" name="sparse_attention" type="BlockSparseAttention" version="extension">
        <data compress_ratio="4"/>
        <input>
            <port id="0" precision="BF16"> <!-- query -->
                <dim>1</dim>    <!-- N -->
                <dim>24</dim>   <!-- H -->
                <dim>-1</dim>   <!-- L -->
                <dim>256</dim>  <!-- E -->
            </port>
            <port id="1" precision="BF16"> <!-- key -->
                <dim>1</dim>    <!-- N -->
                <dim>2</dim>    <!-- Hkv -->
                <dim>-1</dim>   <!-- S -->
                <dim>256</dim>  <!-- E -->
            </port>
            <port id="2" precision="BF16"> <!-- value -->
                <dim>1</dim>    <!-- N -->
                <dim>2</dim>    <!-- Hkv -->
                <dim>-1</dim>   <!-- S -->
                <dim>256</dim>  <!-- Ev -->
            </port>
            <port id="3" precision="I32"> <!-- block_indices from SparseAttentionIndexer -->
                <dim>1</dim>    <!-- N -->
                <dim>1</dim>    <!-- Hs -->
                <dim>-1</dim>   <!-- L -->
                <dim>512</dim>  <!-- KB -->
            </port>
            <port id="4" precision="I32"> <!-- block_count from SparseAttentionIndexer -->
                <dim>1</dim>    <!-- N -->
                <dim>1</dim>    <!-- Hs -->
                <dim>-1</dim>   <!-- L -->
            </port>
        </input>
        <output>
            <port id="5" precision="BF16">
                <dim>1</dim>    <!-- N -->
                <dim>24</dim>   <!-- H -->
                <dim>-1</dim>   <!-- L -->
                <dim>256</dim>  <!-- Ev -->
            </port>
        </output>
    </layer>

*Example 2: One decode step over 10000 cached tokens with an explicit scale*

.. code-block:: xml
   :force:

    <layer id="4" name="sparse_attention" type="BlockSparseAttention" version="extension">
        <data compress_ratio="4"/>
        <input>
            <port id="0" precision="FP32"> <!-- query -->
                <dim>2</dim>     <!-- N -->
                <dim>24</dim>    <!-- H -->
                <dim>1</dim>     <!-- L -->
                <dim>256</dim>   <!-- E -->
            </port>
            <port id="1" precision="FP32"> <!-- key -->
                <dim>2</dim>     <!-- N -->
                <dim>2</dim>     <!-- Hkv -->
                <dim>10001</dim> <!-- S -->
                <dim>256</dim>   <!-- E -->
            </port>
            <port id="2" precision="FP32"> <!-- value -->
                <dim>2</dim>     <!-- N -->
                <dim>2</dim>     <!-- Hkv -->
                <dim>10001</dim> <!-- S -->
                <dim>256</dim>   <!-- Ev -->
            </port>
            <port id="3" precision="I32"> <!-- block_indices from SparseAttentionIndexer -->
                <dim>2</dim>     <!-- N -->
                <dim>1</dim>     <!-- Hs -->
                <dim>1</dim>     <!-- L -->
                <dim>512</dim>   <!-- KB -->
            </port>
            <port id="4" precision="I32"> <!-- block_count from SparseAttentionIndexer: 512 -->
                <dim>2</dim>     <!-- N -->
                <dim>1</dim>     <!-- Hs -->
                <dim>1</dim>     <!-- L -->
            </port>
            <port id="5" precision="FP32"/> <!-- scale -->
        </input>
        <output>
            <!-- each query attends to 512 blocks x 4 tokens + 1 token of its incomplete block = 2049 of the 10001 tokens -->
            <port id="6" precision="FP32">
                <dim>2</dim>     <!-- N -->
                <dim>24</dim>    <!-- H -->
                <dim>1</dim>     <!-- L -->
                <dim>256</dim>   <!-- Ev -->
            </port>
        </output>
    </layer>
