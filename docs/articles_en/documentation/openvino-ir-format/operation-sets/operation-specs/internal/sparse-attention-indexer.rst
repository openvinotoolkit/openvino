SparseAttentionIndexer
======================


.. meta::
  :description: Learn about SparseAttentionIndexer - the indexer of dynamic sparse attention that
                selects the key blocks each query attends to (Qwen Sparse Attention, DeepSeek Sparse
                Attention).

**Versioned name**: *SparseAttentionIndexer*

**Category**: *Sequence processing*

**Short description**: *SparseAttentionIndexer* scores the complete key blocks visible to every
query token and returns the indices of a budget of the best-scoring blocks together with their
number.

**Detailed description**:

*SparseAttentionIndexer* is the selection stage of dynamic, query-dependent sparse attention. Its
output is consumed by :doc:`BlockSparseAttention <block-sparse-attention>`. It covers:

* Qwen Sparse Attention (QSA) of the attention layers of Qwen3.8-Flash-Next (``qwen4_exp`` in
  Hugging Face Transformers), with blocks of ``compress_ratio = 4`` tokens. Most of this document
  uses it as the example;
* DeepSeek Sparse Attention (DSA) of DeepSeek-V3.2 and GLM-MoE-DSA, with single-token blocks
  (``compress_ratio = 1``). See *Relation to DeepSeek Sparse Attention (DSA)* below.

The selection is decoupled from the attention in the same way as in inference engines:

* vLLM (``vllm/models/qwen4_exp``): its ``QSAIndexer`` class scores a compressed key cache with one
  key per complete block and selects the top blocks into a per-query buffer of block indices. A
  separate step expands them into token indices for the sparse attention kernel.
* llama.cpp (``src/models/qwen4exp.cpp``): ``build_qsa_top_k`` scores mean-pooled block keys and
  takes the top-k, which ``build_attn_qsa`` consumes.

Unlike the engines, which expand the selected blocks into token positions before the attention,
*SparseAttentionIndexer* returns block indices. Expanding the blocks and adding the tokens of the
query's incomplete block is done by *BlockSparseAttention*. This keeps the selection
``compress_ratio`` times smaller and lets the attention read every selected block as a contiguous
run of tokens.

The selection works on blocks of ``compress_ratio`` consecutive key tokens: block ``b`` holds the
positions ``b * compress_ratio .. b * compress_ratio + compress_ratio - 1``. Each complete block is
represented by one index key (``index_key_blocks``). For every query token,
*SparseAttentionIndexer*:

1. Scores every complete block that ends at or before the query position with the multi-head,
   ReLU-rectified dot product of the index query heads and the block key, summed over the heads
   (the "lightning indexer" scoring of DeepSeek sparse attention, applied to block keys). The
   optional ``index_weights`` weight every head per query token before the sum; Qwen3.8-Flash-Next
   does not use them, which is equivalent to all weights equal to ``1``.
2. Selects the ``token_budget / compress_ratio`` best-scoring blocks, or all of them if there are
   fewer.

The operation is weight-free: the index projection, normalizations and rotary embeddings are
computed by other operations and passed in through ``index_query`` and ``index_key_blocks``.

**Selection heads.** By default a query token gets one selection, shared by all attention heads that
consume it (``num_selection_heads = 1``); this is the case of QSA and DSA, where the scores of all
``Hi`` index query heads are summed into one score per block. With ``num_selection_heads = Hs > 1``
the ``Hi`` index query heads are split into ``Hs`` consecutive groups of ``Hi / Hs`` heads; the scores
are summed within each group only, and every group selects its own blocks. The outputs then hold
``Hs`` selections per query token, and *BlockSparseAttention* applies selection ``g`` to the group
``g`` of ``H / Hs`` consecutive attention query heads. For example, ``Hs = Hkv`` gives one selection
per key-value head shared by its group of query heads, the granularity of Native Sparse Attention
(NSA); ``Hs = H`` gives one selection per attention query head. ``Hs`` is the number of selections
per token and is independent of ``Hi``, the number of index query heads that score: with ``Hs = 1``
any number of index heads is reduced to one selection.

*SparseAttentionIndexer* provides functionality according to the following pseudo-code using ``numpy``:

.. code-block:: py
   :force:

    def SparseAttentionIndexer(index_query, index_key_blocks, key_length, index_weights=None, *,
                               compress_ratio, token_budget, num_selection_heads=1):
        N, Hi, L, Di = index_query.shape
        S = key_length
        r = compress_ratio
        Hs = num_selection_heads
        G = Hi // Hs                                            # index query heads per selection head
        block_topk = token_budget // r
        if index_weights is None:
            index_weights = numpy.ones((N, L, Hi))

        block_indices = numpy.full((N, Hs, L, block_topk), -1)
        block_count = numpy.zeros((N, Hs, L))
        for n in range(N):
            for g in range(Hs):
                heads = slice(g * G, (g + 1) * G)               # the index query heads of selection head g
                for l in range(L):
                    p = S - L + l                               # position of the query token
                    num_visible_blocks = (p + 1) // r           # complete blocks at or before p
                    if num_visible_blocks == 0:
                        continue
                    q = index_query[n, heads, l, :].astype(numpy.float32)                 # [G, Di]
                    k = index_key_blocks[n, :num_visible_blocks, :].astype(numpy.float32) # [B, Di]
                    w = index_weights[n, l, heads].astype(numpy.float32)                  # [G]
                    scores = (w[:, None] * numpy.maximum(q @ k.T, 0.0)).sum(axis=0)      # [B]
                    num_selected_blocks = min(block_topk, num_visible_blocks)
                    # stable top-k: descending score, equal scores -> smaller block index first
                    blocks = numpy.argsort(-scores, kind="stable")[:num_selected_blocks]
                    block_indices[n, g, l, :num_selected_blocks] = blocks
                    block_count[n, g, l] = num_selected_blocks
        return block_indices, block_count

Properties that follow from the definition:

* The ``L`` query tokens are the last ``L`` of the ``S`` key tokens: query ``l`` has the position
  ``S - L + l``. ``L == S`` is a prefill without past; ``L == 1`` is a decode step.
* A row of ``block_indices`` is the selection of one selection head for one query token. It holds
  ``min(token_budget / compress_ratio, (p + 1) // compress_ratio)`` valid block indices for the
  query position ``p``, followed by ``-1`` padding. Rows of the first ``compress_ratio - 1``
  positions have no valid entries.
* ``block_count`` holds the number of valid entries of every row of ``block_indices`` and is passed
  to the ``block_count`` input of *BlockSparseAttention*. The valid
  entries are the first ``block_count`` ones, so a consumer can use it as a loop bound instead of
  scanning for ``-1``, like the trailing count column of the vLLM selection buffer. In a batch
  aligned to the right the count depends only on the query position, so it is the same for every
  sequence of the batch and every selection head.
* The selection heads differ only in which blocks they select, not in how many: all of them score
  the same visible blocks with their own group of index query heads.
* Every valid block index is in ``[0, (p + 1) // compress_ratio)``: all tokens of a selected block
  are at or before ``p``, so the selection is causal. The indices in a row are unique.
* If ``(p + 1) // compress_ratio <= token_budget / compress_ratio``, all complete blocks are
  selected, and the downstream attention is identical to dense causal attention. With the
  Qwen3.8-Flash-Next budget of ``2048`` this holds for the first ``2051`` positions.
* When ``p + 1`` is a multiple of ``compress_ratio``, the block that contains the query is complete
  and competes in the selection like any other block. If it is not selected, the query does not
  attend to itself.
* The scores are only used for ranking. Scaling them by a positive constant (the reference model
  divides them by ``sqrt(Di)``) does not change the result. A positive constant factor can be put
  into ``index_weights`` or omitted.
* ``index_weights`` may be negative. They cannot be folded into ``index_query``, because
  ``w * ReLU(q . k)`` differs from ``ReLU((w * q) . k)`` for ``w < 0``.
* The valid block indices of a row are ordered by descending score, equal scores by ascending
  block index. The order does not affect the result of *BlockSparseAttention*.

.. note::

   **Equal scores.** In case of score ties, blocks with smaller block indices take precedence
   (stable top-k). The selected blocks are the first ``block_count`` blocks in the order of
   descending score and, for equal scores, ascending block index. When blocks with equal scores
   compete for the last selected places, the blocks with the smaller indices are selected, so the
   selection is deterministic and the same in every invocation and on every device.

**Preparing the inputs** (informative)

In Qwen3.8-Flash-Next both inputs come from one projection of the attention layer input,
``index_qk_proj``, that produces ``Hi`` query heads and one key head of size ``Di``:

* ``index_query``: the query heads, RMS-normalized with ``q_layernorm`` and rotated with the same
  (M-)RoPE and positions as the main attention query.
* ``index_key_blocks``: the raw key head of every token (before normalization and rotation) is
  mean-pooled over each complete block of ``compress_ratio`` tokens, RMS-normalized with
  ``k_layernorm``, and rotated with the (M-)RoPE position of the first token of the block.

In a stateful model with *ScaledDotProductAttention*-style caches, the raw index keys of the past
tokens are kept as one more state next to the key and value caches, with the shape ``[N, S, Di]``,
and ``index_key_blocks`` is recomputed from it. Because a block key depends only on the tokens of
its own block, an implementation may instead keep the compressed keys of the complete blocks plus
the raw keys of at most ``compress_ratio - 1`` tokens of the incomplete block, as vLLM does.

**Relation to DeepSeek Sparse Attention (DSA)**

.. note::

   With ``compress_ratio = 1`` and ``index_weights`` provided, *SparseAttentionIndexer* covers the DSA indexer
   of DeepSeek-V3.2 and GLM-MoE-DSA (``DeepseekV32Indexer`` in Hugging Face Transformers):

   * every block is a single token, so a query at the position ``p`` scores all tokens
     ``0 .. p``, including itself, and selects ``min(token_budget, p + 1)`` of them. There is no
     incomplete block;
   * ``token_budget`` is ``index_topk`` (``2048``), ``Hi`` is ``index_n_heads`` (``64``),
     ``Di`` is ``index_head_dim`` (``128``) and ``num_selection_heads`` is ``1``: the weighted
     scores of all index heads are summed into one selection per token;
   * ``index_key_blocks`` is the per-token index key ``k_norm(wk(x))`` with the partial RoPE of
     the indexer, that is the DSA indexer key cache. Mean-pooling over one token is the identity;
   * ``index_query`` is ``wq_b(q_resid)`` with the same partial RoPE;
   * ``index_weights`` is ``weights_proj(x) * index_n_heads^-0.5``. The positive factor
     ``softmax_scale = index_head_dim^-0.5`` of the DSA score does not change the ranking and can
     be omitted or multiplied into ``index_weights``.

   DSA scores the keys with ``sum_h w[h] * ReLU(softmax_scale * q[h] . k)``, which is the score of
   *SparseAttentionIndexer* up to the positive factor. The Hadamard rotation and the FP8 quantization of the
   reference DSA indexer are precision optimizations and are not part of the definition. The
   selected tokens are consumed by :doc:`BlockSparseAttention <block-sparse-attention>` with
   ``compress_ratio = 1``.


**Attributes**

* *compress_ratio*

  * **Description**: number of consecutive key tokens that form one block.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *token_budget*

  * **Description**: maximum number of tokens selected from complete blocks for one query.
    ``token_budget / compress_ratio`` blocks are selected.
  * **Range of values**: a positive integer divisible by *compress_ratio*
  * **Type**: ``int``
  * **Required**: *yes*

* *num_selection_heads*

  * **Description**: the number of selections per query token, ``Hs``. The ``Hi`` index query heads
    are split into ``Hs`` consecutive groups, the scores are summed within each group, and every
    group selects its own blocks. ``1`` for QSA (Qwen3.8-Flash-Next) and DSA.
  * **Range of values**: a positive integer that divides ``Hi`` and the number of attention query
    heads ``H`` of the consuming *BlockSparseAttention*
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*

* *index_element_type*

  * **Description**: the element type of the ``block_indices`` and ``block_count`` outputs.
  * **Range of values**: ``i32``, ``i64``
  * **Type**: ``string``
  * **Default value**: ``i32``
  * **Required**: *no*


**Inputs**

* **1**: ``index_query`` - 4D tensor of type *T* and shape ``[N, Hi, L, Di]``: the index query
  heads of the query tokens. **Required.**

* **2**: ``index_key_blocks`` - 3D tensor of type *T* and shape ``[N, B, Di]``: one index key per
  complete key block, ``B = floor(S / compress_ratio)``. **Required.**

* **3**: ``key_length`` - a scalar or single element 1D tensor of type *T_IND*: the number of key
  tokens ``S``, including the query tokens. ``S >= L``. **Required.**

* **4**: ``index_weights`` - 3D tensor of type *T* and shape ``[N, L, Hi]``: the weight of every
  index query head for every query token, applied to the rectified head scores before they are
  summed. If not provided, all weights are ``1``, as in Qwen3.8-Flash-Next. DSA models provide
  them, see the note above. **Optional.**


**Outputs**

* **1**: ``block_indices`` - 4D tensor of type *index_element_type* and shape
  ``[N, Hs, L, token_budget / compress_ratio]``: the indices of the key blocks selected by every
  selection head for every query token, padded with ``-1``.

* **2**: ``block_count`` - 3D tensor of type *index_element_type* and shape ``[N, Hs, L]``: the
  number of blocks selected by every selection head for every query token, that is the number of
  valid entries in the corresponding row of ``block_indices``,
  ``min(token_budget / compress_ratio, (p + 1) // compress_ratio)`` for the query position ``p``.


**Types**

* *T*: any supported floating-point type. The scores are accumulated with at least ``f32``
  precision.

* *T_IND*: ``int32`` or ``int64``.


**Dimensions**

* ``N`` - batch size. The sequences of a batch are aligned to the right: query ``l`` of every
  sequence is at position ``S - L + l``. Batches with padding are not supported.

* ``L`` - number of query tokens.

* ``S`` - number of key tokens, given by ``key_length``.

* ``Hi`` - number of index query heads. The indexer has one shared key head.

* ``Hs`` - number of selection heads, *num_selection_heads*. ``Hs`` divides ``Hi``.

* ``Di`` - head size of the index query and key.

* ``B`` - number of complete key blocks, ``floor(S / compress_ratio)``.


**Examples**

*Example 1: Prefill with the Qwen3.8-Flash-Next configuration*

.. code-block:: xml
   :force:

    <layer id="1" name="indexer" type="SparseAttentionIndexer" version="extension">
        <data compress_ratio="4" token_budget="2048" index_element_type="i32"/>
        <input>
            <port id="0" precision="BF16"> <!-- index_query -->
                <dim>1</dim>   <!-- N -->
                <dim>4</dim>   <!-- Hi -->
                <dim>-1</dim>  <!-- L -->
                <dim>128</dim> <!-- Di -->
            </port>
            <port id="1" precision="BF16"> <!-- index_key_blocks -->
                <dim>1</dim>   <!-- N -->
                <dim>-1</dim>  <!-- B = floor(S / 4) -->
                <dim>128</dim> <!-- Di -->
            </port>
            <port id="2" precision="I64"/> <!-- key_length -->
        </input>
        <output>
            <port id="3" precision="I32"> <!-- block_indices -->
                <dim>1</dim>   <!-- N -->
                <dim>1</dim>   <!-- Hs = num_selection_heads -->
                <dim>-1</dim>  <!-- L -->
                <dim>512</dim> <!-- token_budget / compress_ratio -->
            </port>
            <port id="4" precision="I32"> <!-- block_count -->
                <dim>1</dim>   <!-- N -->
                <dim>1</dim>   <!-- Hs = num_selection_heads -->
                <dim>-1</dim>  <!-- L -->
            </port>
        </output>
    </layer>

*Example 2: One decode step over 10000 cached tokens*

.. code-block:: xml
   :force:

    <layer id="2" name="indexer" type="SparseAttentionIndexer" version="extension">
        <data compress_ratio="4" token_budget="2048"/>
        <input>
            <port id="0" precision="FP32"> <!-- index_query -->
                <dim>2</dim>    <!-- N -->
                <dim>4</dim>    <!-- Hi -->
                <dim>1</dim>    <!-- L -->
                <dim>128</dim>  <!-- Di -->
            </port>
            <port id="1" precision="FP32"> <!-- index_key_blocks -->
                <dim>2</dim>    <!-- N -->
                <dim>2500</dim> <!-- B = floor(10001 / 4) -->
                <dim>128</dim>  <!-- Di -->
            </port>
            <port id="2" precision="I32"/> <!-- key_length = 10001 -->
        </input>
        <output>
            <!-- each row: the 512 best of the 2500 visible blocks, no padding -->
            <port id="3" precision="I32"> <!-- block_indices -->
                <dim>2</dim>    <!-- N -->
                <dim>1</dim>    <!-- Hs = num_selection_heads -->
                <dim>1</dim>    <!-- L -->
                <dim>512</dim>  <!-- token_budget / compress_ratio -->
            </port>
            <port id="4" precision="I32"> <!-- block_count: 512 for both sequences -->
                <dim>2</dim>    <!-- N -->
                <dim>1</dim>    <!-- Hs = num_selection_heads -->
                <dim>1</dim>    <!-- L -->
            </port>
        </output>
    </layer>

*Example 3: DSA (DeepSeek-V3.2) indexer, one decode step over 10000 cached tokens*

.. code-block:: xml
   :force:

    <layer id="3" name="dsa_indexer" type="SparseAttentionIndexer" version="extension">
        <data compress_ratio="1" token_budget="2048"/>
        <input>
            <port id="0" precision="BF16"> <!-- index_query -->
                <dim>1</dim>     <!-- N -->
                <dim>64</dim>    <!-- Hi = index_n_heads -->
                <dim>1</dim>     <!-- L -->
                <dim>128</dim>   <!-- Di = index_head_dim -->
            </port>
            <port id="1" precision="BF16"> <!-- index_key_blocks: per-token index keys -->
                <dim>1</dim>     <!-- N -->
                <dim>10001</dim> <!-- B = S -->
                <dim>128</dim>   <!-- Di -->
            </port>
            <port id="2" precision="I32"/> <!-- key_length = 10001 -->
            <port id="3" precision="BF16"> <!-- index_weights -->
                <dim>1</dim>     <!-- N -->
                <dim>1</dim>     <!-- L -->
                <dim>64</dim>    <!-- Hi -->
            </port>
        </input>
        <output>
            <port id="4" precision="I32"> <!-- block_indices: token indices -->
                <dim>1</dim>     <!-- N -->
                <dim>1</dim>     <!-- Hs = num_selection_heads -->
                <dim>1</dim>     <!-- L -->
                <dim>2048</dim>  <!-- token_budget / compress_ratio = index_topk -->
            </port>
            <port id="5" precision="I32"> <!-- block_count: 2048 -->
                <dim>1</dim>     <!-- N -->
                <dim>1</dim>     <!-- Hs = num_selection_heads -->
                <dim>1</dim>     <!-- L -->
            </port>
        </output>
    </layer>
