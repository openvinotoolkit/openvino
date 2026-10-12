BlockSparseAttention
====================


.. meta::
  :description: Learn about BlockSparseAttention - grouped-query scaled dot-product attention
                restricted to dynamically selected key blocks, optionally causal with the query's
                incomplete block, with selections per query token or per query block.

**Versioned name**: *BlockSparseAttention*

**Category**: *Sequence processing*

**Short description**: *BlockSparseAttention* computes grouped-query scaled dot-product attention
in which every query token attends only to the tokens of its selected key blocks and, in the causal
mode, to the tokens of the incomplete block that contains it.

**Detailed description**:

*BlockSparseAttention* is the attention stage of dynamic, query-dependent sparse attention. The
blocks are not a fixed sparsity pattern: they are selected per query token, or per block of
``query_block_size`` query tokens, and passed in through ``block_indices``. It covers:

* Qwen Sparse Attention (QSA) of the attention layers of Qwen3.8-Flash-Next (``qwen4_exp`` in
  Hugging Face Transformers), with blocks of ``compress_ratio = 4`` tokens selected by
  :doc:`QSAIndexer <qsa-indexer>`. Most of this document uses it as the
  example;
* DeepSeek Sparse Attention (DSA) of DeepSeek-V3.2 and GLM-MoE-DSA, with single-token blocks
  (``compress_ratio = 1``). See the note below;
* Locality-Constrained Sparse Attention (LCSA) of the FlashVSR video super-resolution model, which
  is not causal and selects key blocks of 128 tokens per head for every block of 128 query tokens
  (``causal = false``, ``compress_ratio = 128``, ``query_block_size = 128``, ``Hs = H``). See the
  note *FlashVSR* below.

The decoupling from the selection follows inference engines:

* vLLM (``vllm/models/qwen4_exp``): the sparse paged attention kernel attends to the tokens of the
  blocks selected by its ``QSAIndexer`` class plus the incomplete block of the query, read from the
  main key/value cache.
* llama.cpp (``src/models/qwen4exp.cpp``): ``build_attn_qsa`` starts from a mask filled with
  ``-inf``, writes ``0`` at the selected positions with ``ggml_set_rows``, and runs regular
  multi-head attention with that mask.

Key blocks consist of ``compress_ratio`` consecutive key tokens: block ``b`` holds the positions
``b * compress_ratio .. b * compress_ratio + compress_ratio - 1``. Query tokens share selections in
blocks of ``query_block_size``: query token ``l`` uses the row ``l // query_block_size`` of
``block_indices`` and ``block_count``. With ``query_block_size = 1``, the default, every query token
has its own row. A query token attends to:

* all tokens of the first ``block_count`` blocks listed in its row of ``block_indices`` for the
  selection head of the attention head;
* only in the causal mode (``causal = true``, the default): the tokens of the incomplete block that
  contains it, ``floor((p + 1) / compress_ratio) * compress_ratio .. p`` for its position ``p``: at
  most ``compress_ratio - 1`` tokens, including the query itself. This part is empty when ``p + 1``
  is a multiple of ``compress_ratio``, because the block of the query is then complete and attended
  only if it is listed in ``block_indices``.

In the causal mode the query tokens are the last ``L`` of the ``S`` key tokens, at the positions
``S - L + l``, and the selected blocks must be complete blocks at or before the query. With
``causal = false`` there is no position rule and no incomplete block: a query token attends to
exactly the tokens of its selected blocks, which may be anywhere in the key sequence, the queries
need not be a part of the keys, and the last key block may be partial.

**Selection heads.** ``block_indices`` and ``block_count`` hold ``Hs`` selections per query token.
The ``H`` attention query heads are split into ``Hs`` consecutive groups of ``H / Hs`` heads, and
query head ``h`` uses the selection ``h // (H / Hs)``. With ``Hs = 1`` all query heads of a token
share one selection, the case of QSA and DSA; with ``Hs = Hkv`` every key-value head and its group
of query heads has its own selection, as in Native Sparse Attention (NSA); with ``Hs = H`` every
query head has its own selection. In the causal mode, the incomplete block of the query is attended
by every head.

There is no other mask: in the causal mode the causality comes from the selection, and with
``causal = false`` the selection alone defines the attended tokens. This document covers the
*ScaledDotProductAttention* case: the key and
value tensors hold all tokens of the sequences contiguously, as with a stateful KV cache, rather
than in a paged cache.

*BlockSparseAttention* provides functionality according to the following pseudo-code using
``numpy``:

.. code-block:: py
   :force:

    def BlockSparseAttention(query, key, value, block_indices, block_count, scale=None, *,
                             compress_ratio, query_block_size=1, causal=True):
        N, H, L, E = query.shape
        Hkv, S = key.shape[1], key.shape[2]
        Hs = block_indices.shape[1]                             # selection heads
        r, qb = compress_ratio, query_block_size
        if scale is None:
            scale = 1.0 / sqrt(E)

        mask = numpy.zeros((N, Hs, L, S), dtype=bool)
        for n in range(N):
            for g in range(Hs):
                for l in range(L):
                    row = l // qb                               # the selection row of query token l
                    blocks = block_indices[n, g, row, :int(block_count[n, g, row])]  # the rest is ignored
                    selected = (blocks[:, None] * r + numpy.arange(r)[None, :]).reshape(-1)
                    mask[n, g, l, selected[selected < S]] = True        # the last key block may be partial
                    if causal:
                        p = S - L + l                           # position of the query token
                        tail = numpy.arange((p + 1) // r * r, p + 1)    # incomplete block, < r tokens
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

* Causal mode: the ``L`` query tokens are the last ``L`` of the ``S`` key tokens, and query ``l``
  has the position ``S - L + l``. With the valid block indices of a row in
  ``[0, (p + 1) // compress_ratio)``, as produced by *QSAIndexer*, every attended token
  is at or before ``p``. A query attends to ``block_count * compress_ratio + (p + 1) % compress_ratio``
  tokens, so the number of key and value rows of every query is known before any of them is read.
* With ``causal = false`` a query attends to the ``block_count * compress_ratio`` tokens of its
  selected blocks, fewer if the last, partial key block is selected.
* All ``query_block_size`` query tokens of a query block attend to the same selected key blocks, so
  an implementation can compute every pair of a selected key block and a query block as one tile,
  as the 128 x 128 blocks of the Block-Sparse-Attention library used by FlashVSR.
* ``block_count`` is the loop bound over the selected blocks of a row: an implementation reads only
  the first ``block_count`` entries and never scans the row for ``-1``. It can also size the work of
  a query and skip rows without selected blocks without reading ``block_indices``, like the valid-entry
  count column that the vLLM sparse attention kernel uses as its tile-loop bound.
* The result does not depend on the order of the valid block indices in a row or on the entries
  after the first ``block_count`` ones, which are ignored.
* With the rows produced by *QSAIndexer*, the result is identical to causal
  *ScaledDotProductAttention* as long as every complete block is selected, that is for the first
  ``token_budget + compress_ratio - 1`` positions.
* A query head attends to nothing, and produces zeros, if the ``block_count`` of its selection is
  ``0`` and, in the causal mode, ``p + 1`` is a multiple of ``compress_ratio``.
  *QSAIndexer* never produces such a row; the FlashVSR selection can.
* With ``Hs = 1`` every query head of a token attends to the same key and value rows, so an
  implementation can read every selected block once for all heads, as for multi-query attention.
  With ``Hs = Hkv`` the rows are shared within every group of query heads of one key-value head.
* The operation can be expressed with other operations by expanding ``block_indices`` and, in the
  causal mode, the incomplete block into a boolean mask and passing it to
  *ScaledDotProductAttention* as ``attention_mask``, as llama.cpp does. That costs time and memory
  proportional to ``S`` for every query, while a dedicated implementation reads only the selected
  blocks, each as a contiguous run of ``compress_ratio`` key and value rows.

The sigmoid output gate of the Qwen3.8-Flash-Next attention is applied to the output by a separate
*Multiply* and is not part of this operation.

.. note::

   With ``compress_ratio = 1``, every block is a single token and the incomplete block is always
   empty, so a query attends to exactly the tokens listed in ``block_indices``. Together with
   :doc:`QSAIndexer <qsa-indexer>` configured as described there, this covers the sparse attention
   of DeepSeek Sparse Attention (DSA) models such as DeepSeek-V3.2 and GLM-MoE-DSA. Their
   multi-head latent attention maps to this operation in its absorbed form, as multi-query
   attention with ``Hkv = 1``, ``E = kv_lora_rank + qk_rope_head_dim`` (``576``) and
   ``Ev = kv_lora_rank`` (``512``), with the attention scale passed through ``scale``.

.. note::

   **FlashVSR.** The Locality-Constrained Sparse Attention of FlashVSR (a Wan2.1-1.3B diffusion
   transformer with 12 heads of 128; ``JunhaoZhuang/FlashVSR``, code in ``OpenImagingLab/FlashVSR``)
   maps to this operation with ``causal = false``, ``compress_ratio = 128``,
   ``query_block_size = 128`` and ``Hs = H``:

   * the graph reorders the tokens into 3D windows of 2 latent frames x 8 x 8 tokens
     (``WindowPartition3D``), so that every window is a block of 128 consecutive tokens. The queries
     are the windows of the current chunk of 2 latent frames, the keys and values are the windows
     of the cached chunks and the current chunk;
   * the selection is built by ordinary operations in the graph, as in FlashVSR's
     ``generate_draft_block_mask``: mean-pooled queries and keys per window, their scaled dot
     product plus a ``-inf`` mask outside a ``local_range x local_range`` neighborhood of windows,
     a softmax over the key windows, and a threshold ``t`` equal to the ``(topk + 1)``-th largest
     probability of every head and query chunk. FlashVSR attends to the key windows with a
     probability strictly greater than ``t``;
   * the boolean selection is passed as indices and counts, without a dense block mask:
     ``block_indices = TopK(p, k = number of key windows, axis = -1, mode = max, sort = value)`` and
     ``block_count = ReduceSum(Greater(p, t), axis = -1)``. The windows of a row that exceed ``t``
     are its largest entries, and entries equal to ``t`` sort after them, so the first
     ``block_count`` indices are exactly the selected windows. ``KB`` is the number of key windows,
     because a row may select all of them;
   * the number of selected windows differs between the rows and may be ``0``. Such a query block
     produces zeros by this definition; an implementation that must match the Block-Sparse-Attention
     kernel used by FlashVSR should verify the kernel's result for an empty row.


**Attributes**

* *compress_ratio*

  * **Description**: number of consecutive key tokens that form one block. It must be equal to
    *compress_ratio* of the *QSAIndexer* that produces ``block_indices``. ``128`` for
    FlashVSR.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *query_block_size*

  * **Description**: number of consecutive query tokens that share one row of ``block_indices`` and
    ``block_count``: query token ``l`` uses the row ``l // query_block_size``. ``1`` for QSA and DSA,
    ``128`` for FlashVSR.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*

* *causal*

  * **Description**: ``true``: the query tokens are the last ``L`` key tokens, every query token
    also attends to the incomplete block that contains it, and the selected blocks must be complete
    blocks at or before the query (QSA, DSA). ``false``: every query token attends to exactly the
    tokens of its selected blocks, with no position rule (FlashVSR).
  * **Range of values**: ``true``, ``false``
  * **Type**: ``boolean``
  * **Default value**: ``true``
  * **Required**: *no*


**Inputs**

* **1**: ``query`` - 4D tensor of type *T* and shape ``[N, H, L, E]``: the attention query heads.
  **Required.**

* **2**: ``key`` - 4D tensor of type *T* and shape ``[N, Hkv, S, E]``: the attention key heads of
  all tokens, including the past ones. **Required.**

* **3**: ``value`` - 4D tensor of type *T* and shape ``[N, Hkv, S, Ev]``: the attention value heads
  of all tokens, including the past ones. **Required.**

* **4**: ``block_indices`` - 4D tensor of type *T_IND* and shape ``[N, Hs, Lq, KB]``: for every
  selection head and every query row, the indices of the key blocks it attends to in its first
  ``block_count`` entries. These entries are unique within a row and, in the causal mode, in
  ``[0, (p + 1) // compress_ratio)`` for the position ``p`` of every query token of the row; with
  ``causal = false`` they are in ``[0, ceil(S / compress_ratio))``. Otherwise the behavior is
  undefined. The remaining entries are ignored; *QSAIndexer* sets them to ``-1``.
  **Required.**

* **5**: ``block_count`` - 3D tensor of type *T_IND* and shape ``[N, Hs, Lq]``: the number of valid
  entries in every row of ``block_indices``, ``0 <= block_count <= KB``. Normally the
  ``block_count`` output of *QSAIndexer*. **Required.**

* **6**: ``scale`` - a scalar or single element 1D tensor of type *T*: the attention scale factor,
  used instead of the default ``1 / sqrt(E)``. **Optional.**


**Outputs**

* **1**: ``output`` - 4D tensor of type *T* and shape ``[N, H, L, Ev]``: the result of the sparse
  attention.


**Types**

* *T*: any supported floating-point type.

* *T_IND*: ``int32`` or ``int64``.


**Dimensions**

* ``N`` - batch size. In the causal mode the sequences of a batch are aligned to the right: query
  ``l`` of every sequence is at position ``S - L + l``. Batches with padding are not supported.

* ``H`` - number of attention query heads. ``H`` must be divisible by ``Hkv``.

* ``Hkv`` - number of attention key and value heads.

* ``Hs`` - number of selection heads, given by dimension 1 of ``block_indices`` and ``block_count``.
  ``Hs`` must divide ``H``. ``1`` for QSA and DSA.

* ``L`` - number of query tokens. In the causal mode ``L <= S``.

* ``Lq`` - number of query rows, ``ceil(L / query_block_size)``.

* ``S`` - number of key and value tokens. In the causal mode the query tokens are included.

* ``E`` - head size of the attention query and key.

* ``Ev`` - head size of the attention value.

* ``KB`` - maximum number of selected blocks per query row. For the output of
  *QSAIndexer*, ``KB = token_budget / compress_ratio``; for FlashVSR, the number of key
  windows.


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
            <port id="3" precision="I32"> <!-- block_indices from QSAIndexer -->
                <dim>1</dim>    <!-- N -->
                <dim>1</dim>    <!-- Hs -->
                <dim>-1</dim>   <!-- L -->
                <dim>512</dim>  <!-- KB -->
            </port>
            <port id="4" precision="I32"> <!-- block_count from QSAIndexer -->
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
            <port id="3" precision="I32"> <!-- block_indices from QSAIndexer -->
                <dim>2</dim>     <!-- N -->
                <dim>1</dim>     <!-- Hs -->
                <dim>1</dim>     <!-- L -->
                <dim>512</dim>   <!-- KB -->
            </port>
            <port id="4" precision="I32"> <!-- block_count from QSAIndexer: 512 -->
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

*Example 3: FlashVSR, one streaming step at 768 x 1408 output, not causal*

.. code-block:: xml
   :force:

    <!-- 66 windows of 128 tokens per chunk of 2 latent frames (6 x 11 windows), queries = current chunk,
         keys = 3 cached chunks + the current chunk = 264 windows -->
    <layer id="5" name="lcsa_attention" type="BlockSparseAttention" version="extension">
        <data compress_ratio="128" query_block_size="128" causal="false"/>
        <input>
            <port id="0" precision="BF16"> <!-- query, tokens in 3D-window order -->
                <dim>1</dim>     <!-- N -->
                <dim>12</dim>    <!-- H -->
                <dim>8448</dim>  <!-- L = 66 x 128 -->
                <dim>128</dim>   <!-- E -->
            </port>
            <port id="1" precision="BF16"> <!-- key -->
                <dim>1</dim>     <!-- N -->
                <dim>12</dim>    <!-- Hkv -->
                <dim>33792</dim> <!-- S = 264 x 128 -->
                <dim>128</dim>   <!-- E -->
            </port>
            <port id="2" precision="BF16"> <!-- value -->
                <dim>1</dim>     <!-- N -->
                <dim>12</dim>    <!-- Hkv -->
                <dim>33792</dim> <!-- S -->
                <dim>128</dim>   <!-- Ev -->
            </port>
            <port id="3" precision="I32"> <!-- block_indices = TopK(p).indices -->
                <dim>1</dim>     <!-- N -->
                <dim>12</dim>    <!-- Hs = H -->
                <dim>66</dim>    <!-- Lq = L / 128 -->
                <dim>264</dim>   <!-- KB = number of key windows -->
            </port>
            <port id="4" precision="I32"> <!-- block_count = ReduceSum(p > t), about 120 on average -->
                <dim>1</dim>     <!-- N -->
                <dim>12</dim>    <!-- Hs -->
                <dim>66</dim>    <!-- Lq -->
            </port>
        </input>
        <output>
            <port id="5" precision="BF16">
                <dim>1</dim>     <!-- N -->
                <dim>12</dim>    <!-- H -->
                <dim>8448</dim>  <!-- L -->
                <dim>128</dim>   <!-- Ev -->
            </port>
        </output>
    </layer>
