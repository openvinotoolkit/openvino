SparseAttentionIndexer
======================


.. meta::
  :description: Learn about SparseAttentionIndexer-17 - the indexer of dynamic sparse attention that
                selects the key blocks each query attends to (Qwen Sparse Attention, DeepSeek Sparse
                Attention).

**Versioned name**: *SparseAttentionIndexer-17*

**Category**: *Sequence processing*

**Short description**: *SparseAttentionIndexer* scores the complete key blocks visible to every
query token and returns the indices of a budget of the best-scoring blocks together with their
number.

**Detailed description**:

*SparseAttentionIndexer* is the selection stage of dynamic, query-dependent sparse attention. Its
output is consumed by :doc:`SparseAttention <sparse-attention-17>`. It covers:

* Qwen Sparse Attention (QSA) of the attention layers of Qwen3.8-Flash-Next (``qwen4_exp`` in
  Hugging Face Transformers), with blocks of ``key_block_size = 4`` tokens. Most of this document
  uses it as the example;
* DeepSeek Sparse Attention (DSA) of DeepSeek-V3.2 and GLM-MoE-DSA, with single-token blocks
  (``key_block_size = 1``). See *Relation to DeepSeek Sparse Attention (DSA)* below. The same
  lightning indexer is used by Hunyuan V4 (``hy_v4``) and AXK2;
* the block-level variants of the lightning indexer: GLM-5.3-Flash (``glm5_next``), which pools the
  keys of blocks of ``index_kpool = 16`` tokens (``key_block_size = 16``,
  ``token_budget = index_topk``), and the compressed sparse attention (CSA) layers of DeepSeek-V4,
  which score one compressed entry per ``4`` tokens (``key_block_size = 4``,
  ``token_budget = 4 * index_topk``). Their block keys come from a learned, softmax-gated pooling
  computed by regular operations of the graph instead of
  :doc:`QSAPrepare <../internal/qsa-prepare>`. DeepSeek-V4 attends to the selected compressed entries rather
  than to the tokens of the selected blocks, which *SparseAttention* does not cover.

The selection is decoupled from the attention in the same way as in inference engines:

* vLLM (``vllm/models/qwen4_exp``): its ``QSAIndexer`` class scores a compressed key cache with one
  key per complete block and selects the top blocks into a per-query buffer of block indices. A
  separate step expands them into token indices for the sparse attention kernel.
* llama.cpp (``src/models/qwen4exp.cpp``): ``build_qsa_top_k`` scores mean-pooled block keys and
  takes the top-k, which ``build_attn_qsa`` consumes.

Unlike the engines, which expand the selected blocks into token positions before the attention,
*SparseAttentionIndexer* returns block indices. Expanding the blocks and adding the tokens of the
query's incomplete block is done by *SparseAttention*. This keeps the selection
``key_block_size`` times smaller and lets the attention read every selected block as a contiguous
run of tokens.

The selection works on blocks of ``key_block_size`` consecutive key tokens: block ``b`` holds the
positions ``b * key_block_size .. b * key_block_size + key_block_size - 1``. Each complete block is
represented by one index key (``block_index_keys``). For every query token,
*SparseAttentionIndexer*:

1. Scores every complete block that ends at or before the query position with the multi-head,
   ReLU-rectified dot product of the index query heads and the block key, summed over the heads
   (the "lightning indexer" scoring of DeepSeek sparse attention, applied to block keys). The
   optional ``index_weights`` weight every head per query token before the sum; Qwen3.8-Flash-Next
   does not use them, which is equivalent to all weights equal to ``1``.
2. Selects the ``token_budget / key_block_size`` best-scoring blocks, or all of them if there are
   fewer.

The operation is weight-free: the index projection and the normalization and rotation of the index
query are computed by regular operations of the graph, and the pooling, normalization and rotation
of the block keys by :doc:`QSAPrepare <../internal/qsa-prepare>`. They are passed in through ``index_query`` and
``block_index_keys``.

**Selection heads.** By default a query token gets one selection, shared by all attention heads that
consume it (``num_selection_heads = 1``); this is the case of QSA and DSA, where the scores of all
``Hi`` index query heads are summed into one score per block. With ``num_selection_heads = Hs > 1``
the ``Hi`` index query heads are split into ``Hs`` consecutive groups of ``Hi / Hs`` heads; the scores
are summed within each group only, and every group selects its own blocks. The outputs then hold
``Hs`` selections per query token, and *SparseAttention* applies selection ``g`` to the group
``g`` of ``H / Hs`` consecutive attention query heads. For example, ``Hs = Hkv`` gives one selection
per key-value head shared by its group of query heads, the granularity of Native Sparse Attention
(NSA); ``Hs = H`` gives one selection per attention query head. ``Hs`` is the number of selections
per token and is independent of ``Hi``, the number of index query heads that score: with ``Hs = 1``
any number of index heads is reduced to one selection.

*SparseAttentionIndexer* provides functionality according to the following pseudo-code using ``numpy``:

.. code-block:: py
   :force:

    def SparseAttentionIndexer(index_query, block_index_keys, key_length, index_weights=None, *,
                   key_block_size, token_budget, num_selection_heads=1):
        N, Hi, L, Di = index_query.shape
        key_length = numpy.broadcast_to(key_length, (N,))       # valid key tokens of every sequence
        r = key_block_size
        Hs = num_selection_heads
        G = Hi // Hs                                            # index query heads per selection head
        block_topk = token_budget // r
        if index_weights is None:
            index_weights = numpy.ones((N, L, Hi))

        block_indices = numpy.full((N, Hs, L, block_topk), -1, dtype=numpy.int32)
        block_count = numpy.zeros((N, Hs, L), dtype=numpy.int32)
        for n in range(N):
            # left padding: the first L - key_length[n] query tokens are padding (only in a pre-fill)
            num_padding_queries = max(0, L - key_length[n])
            # the blocks of sequence n are the first rows of block_index_keys[n]; the rows from
            # num_valid_blocks on are fake (padding of the block axis) and are never read below
            num_valid_blocks = key_length[n] // r
            for g in range(Hs):
                heads = slice(g * G, (g + 1) * G)               # the index query heads of selection head g
                for l in range(L):
                    if l < num_padding_queries:                 # padding query: selects no blocks
                        continue
                    p = key_length[n] - L + l                   # position of the query token in its sequence, >= 0
                    num_visible_blocks = (p + 1) // r           # complete blocks at or before p, <= num_valid_blocks
                    if num_visible_blocks == 0:                 # no complete block yet
                        continue
                    q = index_query[n, heads, l, :].astype(numpy.float32)                 # [G, Di]
                    k = block_index_keys[n, :num_visible_blocks, :].astype(numpy.float32) # [B, Di]
                    w = index_weights[n, l, heads].astype(numpy.float32)                  # [G]
                    scores = (w[:, None] * numpy.maximum(q @ k.T, 0.0)).sum(axis=0)      # [B]
                    num_selected_blocks = min(block_topk, num_visible_blocks)
                    # stable top-k: descending score, equal scores -> smaller block index first
                    blocks = numpy.argsort(-scores, kind="stable")[:num_selected_blocks]
                    block_indices[n, g, l, :num_selected_blocks] = blocks
                    block_count[n, g, l] = num_selected_blocks
        return block_indices, block_count

Properties that follow from the definition:

* The ``L`` query tokens are the last ``L`` of the ``key_length[n]`` valid key tokens of sequence
  ``n``: query ``l`` has the position ``p = key_length[n] - L + l`` within its sequence. ``L == S``
  is a prefill without past; ``L == 1`` is a decode step.
* **Batches with padding.** The sequences of a batch are aligned to the right (left padding):
  sequence ``n`` has ``key_length[n]`` valid key tokens, the last ones of the ``S`` key positions,
  and ``S - key_length[n]`` padding tokens before them. The blocks of a sequence start at its first
  valid token, block ``b`` is its ``b``-th block (``block_index_keys[n, b]``, see *QSAPrepare*), and
  the positions and block indices are counted within the sequence, so padding changes nothing for
  the valid tokens: a padded sequence gets the same selection as alone in a batch of one. A query
  that is a padding token (``p < 0``) selects no blocks. Without padding, ``key_length[n] = S`` for
  every sequence.
* A row of ``block_indices`` is the selection of one selection head for one query token. It holds
  ``min(token_budget / key_block_size, (p + 1) // key_block_size)`` valid block indices for the
  query position ``p``, followed by ``-1`` padding. Rows of the first ``key_block_size - 1``
  positions have no valid entries.
* ``block_count`` holds the number of valid entries of every row of ``block_indices`` and is passed
  to the ``block_count`` input of *SparseAttention*. The valid
  entries are the first ``block_count`` ones, so a consumer can use it as a loop bound instead of
  scanning for ``-1``, like the trailing count column of the vLLM selection buffer. The count depends
  only on the query position ``p``, so without padding it is the same for every sequence of the
  batch, and it is the same for every selection head.
* The selection heads differ only in which blocks they select, not in how many: all of them score
  the same visible blocks with their own group of index query heads.
* Every valid block index is in ``[0, (p + 1) // key_block_size)``: all tokens of a selected block
  are at or before ``p``, so the selection is causal. The indices in a row are unique.
* If ``(p + 1) // key_block_size <= token_budget / key_block_size``, all complete blocks are
  selected, and the downstream attention is identical to dense causal attention. With the
  Qwen3.8-Flash-Next budget of ``2048`` this holds for the first ``2051`` positions.
* When ``p + 1`` is a multiple of ``key_block_size``, the block that contains the query is complete
  and competes in the selection like any other block. If it is not selected, the query does not
  attend to itself.
* The scores are only used for ranking. Scaling them by a positive constant (the reference model
  divides them by ``sqrt(Di)``) does not change the result. A positive constant factor can be put
  into ``index_weights`` or omitted.
* ``index_weights`` may be negative. They cannot be folded into ``index_query``, because
  ``w * ReLU(q . k)`` differs from ``ReLU((w * q) . k)`` for ``w < 0``.
* The valid block indices of a row are ordered by descending score, equal scores by ascending
  block index. The order does not affect the result of *SparseAttention*.

.. note::

   **Padding layout: which entries are fake.** With ``pad = S - key_length[n]`` padding tokens of
   sequence ``n``, the token axes and the block axis are padded at opposite ends:

   * **Key token axes** (the raw index keys, the key and value caches) are padded at the
     **beginning**: the key positions ``0 .. pad - 1`` are padding, the valid tokens are at
     ``pad .. S - 1``.
   * **Query axis** (``L`` of ``index_query``): the query tokens are the last ``L`` key positions,
     so the padding queries are also at the **beginning**: query ``l`` is a padding token if
     ``l < L - key_length[n]``, that is ``p < 0``. This happens only in a pre-fill with
     ``L > key_length[n]``; in a decode step every query is valid. A padding query selects no blocks,
     ``block_count = 0``, and its row of ``block_indices`` is ``-1``.
   * **Block axis** (``B`` of ``block_index_keys``) is **left-aligned**: the blocks of a sequence are
     counted from its first valid token, row ``b`` is its ``b``-th block, so the rows
     ``0 .. floor(key_length[n] / key_block_size) - 1`` are valid and the remaining rows at the
     **end** are fake (zeros from *QSAPrepare*). The same holds for the block key cache of
     *QSAPrepare*. The indices in ``block_indices`` are these logical block indices;
     *SparseAttention* maps block ``b`` back to the key positions ``pad + b * key_block_size ..``.
   * The fake rows are never read: a valid query reads only the rows
     ``0 .. (p + 1) // key_block_size - 1``, and ``(p + 1) // key_block_size <=
     floor(key_length[n] / key_block_size)``. Their values therefore do not matter.

   .. code-block:: text
      :force:

      key_block_size = 4, S = 10, key_length = [10, 7]   (pre-fill: L = 10)

      key position:  0   1   2   3   4   5   6   7   8   9
      sequence 0:    a0  a1  a2  a3  a4  a5  a6  a7  a8  a9     pad = 0
      sequence 1:    .   .   .   c0  c1  c2  c3  c4  c5  c6     pad = 3   (. = padding)

      block_index_keys[0] = [ key(a0..a3), key(a4..a7) ]        incomplete block: a8 a9
      block_index_keys[1] = [ key(c0..c3), 0 (fake)    ]        incomplete block: c4 c5 c6
      queries of sequence 1: l = 0, 1, 2 are padding (p = -3, -2, -1); l = 3 is c0 (p = 0)

      next decode step (L = 1, S = 11, key_length = [11, 8]): the new token c7 completes the block
      c4..c7, block 1 of sequence 1 at the key positions 7..10

.. note::

   **Equal scores.** In case of score ties, blocks with smaller block indices take precedence
   (stable top-k). The selected blocks are the first ``block_count`` blocks in the order of
   descending score and, for equal scores, ascending block index. When blocks with equal scores
   compete for the last selected places, the blocks with the smaller indices are selected, so the
   selection is deterministic and the same in every invocation and on every device.

**Preparing the inputs**

Both inputs come from one projection of the attention layer input, ``index_qk_proj`` in
Qwen3.8-Flash-Next, that gives ``Hi`` query heads and one key head of size ``Di``:

* ``index_query``: the query heads, RMS-normalized with ``q_layernorm`` and rotated with the same
  (M-)RoPE and positions as the main attention query. Regular operations of the graph compute it
  (*RMS* with the weight ``1 + w``, *RoPE*), as for the query of the main attention.
* ``block_index_keys``: computed by :doc:`QSAPrepare <../internal/qsa-prepare>`. The raw key head of every token
  (before normalization and rotation) is mean-pooled over each complete block of ``key_block_size``
  tokens, RMS-normalized with ``k_layernorm``, and rotated with the (M-)RoPE position of the first
  token of the block.

The graph flow of a QSA layer is *QSAPrepare* -> *SparseAttentionIndexer* ->
:doc:`SparseAttention <sparse-attention-17>`, with the index query prepared next to
*QSAPrepare* by *RMS* and *RoPE*. In a stateful model with
*ScaledDotProductAttention*-style caches, two more states are kept next to the key and value
caches: the raw index keys of the past tokens, with the shape ``[N, S, Di]``, and the block index
keys of the past complete blocks, with the shape ``[N, B, Di]``, so that *QSAPrepare* computes only
the blocks completed by the new tokens. The paged counterparts are
:doc:`PagedQSAPrepare <../internal/paged-qsa-prepare>` -> :doc:`PagedSparseAttentionIndexer <../internal/paged-sparse-attention-indexer>` ->
:doc:`PagedSparseAttention <../internal/paged-sparse-attention>`.

**Relation to DeepSeek Sparse Attention (DSA)**

.. note::

   With ``key_block_size = 1`` and ``index_weights`` provided, *SparseAttentionIndexer* covers the DSA indexer
   of DeepSeek-V3.2 and GLM-MoE-DSA (``DeepseekV32Indexer`` in Hugging Face Transformers):

   * every block is a single token, so a query at the position ``p`` scores all tokens
     ``0 .. p``, including itself, and selects ``min(token_budget, p + 1)`` of them. There is no
     incomplete block;
   * ``token_budget`` is ``index_topk`` (``2048``), ``Hi`` is ``index_n_heads`` (``64``),
     ``Di`` is ``index_head_dim`` (``128``) and ``num_selection_heads`` is ``1``: the weighted
     scores of all index heads are summed into one selection per token;
   * ``block_index_keys`` is the per-token index key ``k_norm(wk(x))`` with the partial RoPE of
     the indexer, that is the DSA indexer key cache. Mean-pooling over one token is the identity;
   * ``index_query`` is ``wq_b(q_resid)`` with the same partial RoPE;
   * ``index_weights`` is ``weights_proj(x) * index_n_heads^-0.5``. The positive factor
     ``softmax_scale = index_head_dim^-0.5`` of the DSA score does not change the ranking and can
     be omitted or multiplied into ``index_weights``.

   DSA scores the keys with ``sum_h w[h] * ReLU(softmax_scale * q[h] . k)``, which is the score of
   *SparseAttentionIndexer* up to the positive factor. The Hadamard rotation and the FP8 quantization of the
   reference DSA indexer are precision optimizations and are not part of the definition. The
   selected tokens are consumed by :doc:`SparseAttention <sparse-attention-17>` with
   ``key_block_size = 1``.


**Attributes**

* *key_block_size*

  * **Description**: number of consecutive key tokens that form one block.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *token_budget*

  * **Description**: maximum number of tokens selected from complete blocks for one query.
    ``token_budget / key_block_size`` blocks are selected.
  * **Range of values**: a positive integer divisible by *key_block_size*
  * **Type**: ``int``
  * **Required**: *yes*

* *num_selection_heads*

  * **Description**: the number of selections per query token, ``Hs``. The ``Hi`` index query heads
    are split into ``Hs`` consecutive groups, the scores are summed within each group, and every
    group selects its own blocks. ``1`` for QSA (Qwen3.8-Flash-Next) and DSA.
  * **Range of values**: a positive integer that divides ``Hi`` and the number of attention query
    heads ``H`` of the consuming *SparseAttention*
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*


**Inputs**

* **1**: ``index_query`` - 4D tensor of type *T* and shape ``[N, Hi, L, Di]``: the index query
  heads of the query tokens. **Required.**

* **2**: ``block_index_keys`` - 3D tensor of type *T* and shape ``[N, B, Di]``: one index key per
  complete key block of every sequence, ``B = floor(S / key_block_size)``. Row ``n`` holds the keys
  of the ``floor(key_length[n] / key_block_size)`` blocks of sequence ``n`` first; the remaining
  rows of a padded sequence are not read. **Required.**

* **3**: ``key_length`` - 1D tensor of type *T_IND* and shape ``[N]``: the number of valid key tokens
  of every sequence, including its query tokens, ``key_length[n] <= S``. For a batch with padding it
  is the number of non-padding tokens, for example the sum of the 2D attention mask over the key
  positions. A tensor of shape ``[1]`` applies to all sequences; without padding it is ``S``.
  **Required.**

* **4**: ``index_weights`` - 3D tensor of type *T* and shape ``[N, L, Hi]``: the weight of every
  index query head for every query token, applied to the rectified head scores before they are
  summed. If not provided, all weights are ``1``, as in Qwen3.8-Flash-Next. DSA models provide
  them, see the note above. **Optional.**


**Outputs**

* **1**: ``block_indices`` - 4D tensor of type ``int32`` and shape
  ``[N, Hs, L, token_budget / key_block_size]``: the indices of the key blocks selected by every
  selection head for every query token, padded with ``-1``.

* **2**: ``block_count`` - 3D tensor of type ``int32`` and shape ``[N, Hs, L]``: the
  number of blocks selected by every selection head for every query token, that is the number of
  valid entries in the corresponding row of ``block_indices``,
  ``min(token_budget / key_block_size, (p + 1) // key_block_size)`` for the query position ``p``,
  ``0`` for a padding token.


**Types**

* *T*: any supported floating-point type. The scores are accumulated with at least ``f32``
  precision.

* *T_IND*: ``int32`` or ``int64``.


**Dimensions**

* ``N`` - batch size. The sequences of a batch are aligned to the right (left padding), see
  ``key_length``.

* ``L`` - number of query tokens.

* ``S`` - number of key positions, including padding: the key dimension of the attention caches.

* ``Hi`` - number of index query heads. The indexer has one shared key head.

* ``Hs`` - number of selection heads, *num_selection_heads*. ``Hs`` divides ``Hi``.

* ``Di`` - head size of the index query and key.

* ``B`` - number of complete key blocks, ``floor(S / key_block_size)``.


**Examples**

*Example 1: Prefill with the Qwen3.8-Flash-Next configuration*

.. code-block:: xml
   :force:

    <layer id="1" name="indexer" type="SparseAttentionIndexer" version="opset17">
        <data key_block_size="4" token_budget="2048"/>
        <input>
            <port id="0" precision="BF16"> <!-- index_query -->
                <dim>1</dim>   <!-- N -->
                <dim>4</dim>   <!-- Hi -->
                <dim>-1</dim>  <!-- L -->
                <dim>128</dim> <!-- Di -->
            </port>
            <port id="1" precision="BF16"> <!-- block_index_keys -->
                <dim>1</dim>   <!-- N -->
                <dim>-1</dim>  <!-- B = floor(S / 4) -->
                <dim>128</dim> <!-- Di -->
            </port>
            <port id="2" precision="I64"> <dim>1</dim> </port> <!-- key_length = [S] -->
        </input>
        <output>
            <port id="3" precision="I32"> <!-- block_indices -->
                <dim>1</dim>   <!-- N -->
                <dim>1</dim>   <!-- Hs = num_selection_heads -->
                <dim>-1</dim>  <!-- L -->
                <dim>512</dim> <!-- token_budget / key_block_size -->
            </port>
            <port id="4" precision="I32"> <!-- block_count -->
                <dim>1</dim>   <!-- N -->
                <dim>1</dim>   <!-- Hs = num_selection_heads -->
                <dim>-1</dim>  <!-- L -->
            </port>
        </output>
    </layer>

*Example 2: One decode step of a padded batch: 10000 and 9000 cached tokens*

.. code-block:: xml
   :force:

    <layer id="2" name="indexer" type="SparseAttentionIndexer" version="opset17">
        <data key_block_size="4" token_budget="2048"/>
        <input>
            <port id="0" precision="FP32"> <!-- index_query -->
                <dim>2</dim>    <!-- N -->
                <dim>4</dim>    <!-- Hi -->
                <dim>1</dim>    <!-- L -->
                <dim>128</dim>  <!-- Di -->
            </port>
            <port id="1" precision="FP32"> <!-- block_index_keys -->
                <dim>2</dim>    <!-- N -->
                <dim>2500</dim> <!-- B = floor(10001 / 4); sequence 1 uses its first 2250 rows -->
                <dim>128</dim>  <!-- Di -->
            </port>
            <port id="2" precision="I32"> <dim>2</dim> </port> <!-- key_length = [10001, 9001] -->
        </input>
        <output>
            <!-- each row: the 512 best of the 2500 and of the 2250 visible blocks of its sequence -->
            <port id="3" precision="I32"> <!-- block_indices -->
                <dim>2</dim>    <!-- N -->
                <dim>1</dim>    <!-- Hs = num_selection_heads -->
                <dim>1</dim>    <!-- L -->
                <dim>512</dim>  <!-- token_budget / key_block_size -->
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

    <layer id="3" name="dsa_indexer" type="SparseAttentionIndexer" version="opset17">
        <data key_block_size="1" token_budget="2048"/>
        <input>
            <port id="0" precision="BF16"> <!-- index_query -->
                <dim>1</dim>     <!-- N -->
                <dim>64</dim>    <!-- Hi = index_n_heads -->
                <dim>1</dim>     <!-- L -->
                <dim>128</dim>   <!-- Di = index_head_dim -->
            </port>
            <port id="1" precision="BF16"> <!-- block_index_keys: per-token index keys -->
                <dim>1</dim>     <!-- N -->
                <dim>10001</dim> <!-- B = S -->
                <dim>128</dim>   <!-- Di -->
            </port>
            <port id="2" precision="I32"> <dim>1</dim> </port> <!-- key_length = [10001] -->
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
                <dim>2048</dim>  <!-- token_budget / key_block_size = index_topk -->
            </port>
            <port id="5" precision="I32"> <!-- block_count: 2048 -->
                <dim>1</dim>     <!-- N -->
                <dim>1</dim>     <!-- Hs = num_selection_heads -->
                <dim>1</dim>     <!-- L -->
            </port>
        </output>
    </layer>
