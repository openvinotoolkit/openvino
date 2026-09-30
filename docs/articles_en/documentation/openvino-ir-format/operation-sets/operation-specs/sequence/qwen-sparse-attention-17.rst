QwenSparseAttention
===================


.. meta::
  :description: Learn about QwenSparseAttention-17 - a causal attention restricted to the key
                blocks chosen by a lightweight block indexer (Qwen Sparse Attention, QSA).

**Versioned name**: *QwenSparseAttention-17*

**Category**: *Sequence processing*

**Short description**: *QwenSparseAttention* computes causal grouped-query attention where every
query attends only to a budget of whole key blocks selected by a block indexer, plus the incomplete
block that contains the query.

**Detailed description**:

*QwenSparseAttention* implements Qwen Sparse Attention (QSA) used by the attention layers of
Qwen3.8-Flash-Next (``qwen4_exp`` in Hugging Face Transformers). Instead of selecting individual
tokens, QSA works at the level of blocks of ``compress_ratio`` consecutive tokens:

1. **Block scoring.** The key sequence is split into complete blocks of ``compress_ratio`` tokens.
   Each complete block is represented by one compressed index key (``index_key_blocks``). A query
   token scores every complete block that ends at or before it with a multi-head, ReLU-rectified
   dot product against its index query heads (``index_query``), summed over the heads. This is the
   "lightning indexer" scoring of DeepSeek sparse attention applied to block keys.
2. **Block selection.** The ``token_budget / compress_ratio`` best-scoring blocks are selected.
   The tokens of the incomplete block that contains the query (at most ``compress_ratio - 1``
   tokens, up to and including the query itself) are always selected.
3. **Sparse attention.** Scaled dot-product attention with grouped-query heads is computed over the
   selected tokens only. All query heads of a token share the same selection.

The operation is weight-free: all projections, normalizations and rotary embeddings of both the
attention and the indexer are computed by other operations and passed in as inputs. This mirrors
how inference engines split QSA:

* vLLM (``vllm/models/qwen4_exp``): ``QSAIndexer`` keeps a compressed key cache with one normalized,
  rotated key per complete block, scores it, selects the top blocks, and expands them into a
  ``-1``-padded list of ``token_budget + compress_ratio - 1`` token indices per query. The sparse
  paged attention kernel then attends to exactly those tokens.
* llama.cpp (``src/models/qwen4exp.cpp``): ``build_qsa_top_k`` mean-pools the cached raw index keys
  per block, normalizes and rotates them, scores them with ReLU and a head sum, and takes
  ``ggml_top_k`` over ``indexer_top_k + compress_ratio - 1`` positions. ``build_attn_qsa`` turns the
  indices into an additive mask on top of the causal mask and runs regular multi-head attention.

*QwenSparseAttention* provides functionality according to the following pseudo-code using
``numpy``:

.. code-block:: py
   :force:

    def QwenSparseAttention(query, key, value, index_query, index_key_blocks, scale=None, *,
                            compress_ratio, token_budget):
        N, H, L, E = query.shape
        Hkv, S = key.shape[1], key.shape[2]
        r = compress_ratio
        block_topk = token_budget // r
        width = token_budget + r - 1
        if scale is None:
            scale = 1.0 / sqrt(E)

        indices = numpy.full((N, L, width), -1)
        mask = numpy.zeros((N, L, S), dtype=bool)
        for n in range(N):
            for l in range(L):
                p = S - L + l                                   # position of the query token
                num_visible_blocks = (p + 1) // r               # complete blocks at or before p
                selected = numpy.zeros((0,), dtype=numpy.int64)
                if num_visible_blocks > 0:
                    q = index_query[n, :, l, :].astype(numpy.float32)                     # [Hi, Di]
                    k = index_key_blocks[n, :num_visible_blocks, :].astype(numpy.float32) # [B, Di]
                    scores = numpy.maximum(q @ k.T, 0.0).sum(axis=0)                     # [B]
                    num_selected_blocks = min(block_topk, num_visible_blocks)
                    # descending score, equal scores -> smaller block index first
                    blocks = numpy.argsort(-scores, kind="stable")[:num_selected_blocks]
                    selected = (blocks[:, None] * r + numpy.arange(r)[None, :]).reshape(-1)
                tail = numpy.arange(num_visible_blocks * r, p + 1)  # incomplete block, < r tokens
                selected = numpy.concatenate([selected, tail])
                indices[n, l, :selected.size] = selected
                mask[n, l, selected] = True

        key = numpy.repeat(key, H // Hkv, axis=1)
        value = numpy.repeat(value, H // Hkv, axis=1)
        attn = (query @ numpy.swapaxes(key, -1, -2)) * scale        # [N, H, L, S]
        attn = numpy.where(mask[:, None, :, :], attn, -inf)
        attn = Softmax(attn, axis=-1)
        return attn @ value, indices

Properties that follow from the definition:

* The ``L`` query tokens are the last ``L`` tokens of the ``S`` key tokens: query ``l`` has the
  position ``S - L + l``. ``L == S`` is a prefill without past; ``L == 1`` is a decode step.
* Every selected token is at or before the query position, so the selection is causal by
  construction and no separate causal mask is needed.
* A query token attends to at most ``token_budget + compress_ratio - 1`` tokens. If
  ``num_visible_blocks <= token_budget / compress_ratio``, every block is selected and the result
  is identical to causal *ScaledDotProductAttention*. With the Qwen3.8-Flash-Next budget of
  ``2048`` this holds for the first ``2051`` positions.
* When ``p + 1`` is a multiple of ``compress_ratio``, the block that contains the query is complete
  and competes in the block selection like any other block. If it is not selected, the query does
  not attend to itself.
* The index scores are only used for ranking. Scaling them by a positive constant (the reference
  model divides them by ``sqrt(Di)``) does not change the result.
* Selected blocks are ordered by descending score and the incomplete-block tokens follow in
  ascending order. The order does not affect ``output``; it is only visible in ``indices``.

**Preparing the index inputs** (informative)

In Qwen3.8-Flash-Next both index inputs come from one projection of the layer input,
``index_qk_proj``, that produces ``Hi`` query heads and one key head of size ``Di``:

* ``index_query``: the query heads, RMS-normalized with ``q_layernorm`` and rotated with the same
  (M-)RoPE and positions as the main attention query.
* ``index_key_blocks``: the raw key head of every token (before normalization and rotation) is
  mean-pooled over each complete block of ``compress_ratio`` tokens, RMS-normalized with
  ``k_layernorm``, and rotated with the RoPE position of the first token of the block.

For incremental decoding, the raw index keys have to persist between inference calls in addition
to the regular key and value caches. Because a block key depends only on the tokens of its own
block, an implementation may instead keep the compressed block keys, which are final once the block
is complete, plus the raw keys of at most ``compress_ratio - 1`` tokens of the current incomplete
block. vLLM uses this layout: a compressed key cache with one row per complete block, and a circular
buffer of ``compress_ratio`` raw keys per sequence.

The sigmoid output gate of the Qwen3.8-Flash-Next attention is applied to ``output`` by a separate
*Multiply* and is not part of this operation.


**Attributes**

* *compress_ratio*

  * **Description**: number of consecutive key tokens that form one block. Blocks start at
    positions ``0, compress_ratio, 2 * compress_ratio, ...``.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *token_budget*

  * **Description**: maximum number of tokens selected from complete blocks for one query.
    ``token_budget / compress_ratio`` blocks are selected.
  * **Range of values**: a positive integer divisible by *compress_ratio*
  * **Type**: ``int``
  * **Required**: *yes*

* *index_element_type*

  * **Description**: the element type of the ``indices`` output.
  * **Range of values**: ``i32``, ``i64``
  * **Type**: ``string``
  * **Default value**: ``i32``
  * **Required**: *no*


**Inputs**

* **1**: ``query`` - 4D tensor of type *T* and shape ``[N, H, L, E]``: the attention query heads.
  **Required.**

* **2**: ``key`` - 4D tensor of type *T* and shape ``[N, Hkv, S, E]``: the attention key heads of
  all tokens, including the past ones. **Required.**

* **3**: ``value`` - 4D tensor of type *T* and shape ``[N, Hkv, S, Ev]``: the attention value heads
  of all tokens, including the past ones. **Required.**

* **4**: ``index_query`` - 4D tensor of type *T* and shape ``[N, Hi, L, Di]``: the indexer query
  heads of the query tokens. **Required.**

* **5**: ``index_key_blocks`` - 3D tensor of type *T* and shape ``[N, B, Di]``: one indexer key per
  complete block of the key sequence, ``B = floor(S / compress_ratio)``. **Required.**

* **6**: ``scale`` - a scalar or single element 1D tensor of type *T*: the attention scale factor,
  used instead of the default ``1 / sqrt(E)``. It does not apply to the index scores. **Optional.**


**Outputs**

* **1**: ``output`` - 4D tensor of type *T* and shape ``[N, H, L, Ev]``: the result of the sparse
  attention.

* **2**: ``indices`` - 3D tensor of type *index_element_type* and shape
  ``[N, L, token_budget + compress_ratio - 1]``: the positions of the key tokens selected for every
  query token, padded with ``-1``. The row of a query token lists the tokens of its selected blocks
  in descending block-score order, then the tokens of its incomplete block in ascending order.


**Types**

* *T*: any supported floating-point type. The index scores are accumulated with at least ``f32``
  precision.


**Dimensions**

* ``N`` - batch size. The operation assumes the sequences of a batch are aligned to the right:
  query ``l`` of every sequence is at position ``S - L + l`` and the key tokens are at positions
  ``0 .. S - 1``. Batches with padding are not supported.

* ``H`` - number of attention query heads. ``H`` must be divisible by ``Hkv``.

* ``Hkv`` - number of attention key and value heads.

* ``L`` - number of query tokens, ``L <= S``.

* ``S`` - number of key and value tokens, including the query tokens.

* ``E`` - head size of the attention query and key.

* ``Ev`` - head size of the attention value.

* ``Hi`` - number of indexer query heads. The indexer has one shared key head.

* ``Di`` - head size of the indexer query and key.

* ``B`` - number of complete key blocks, ``floor(S / compress_ratio)``.


**Examples**

*Example 1: Prefill with the Qwen3.8-Flash-Next configuration*

.. code-block:: xml
   :force:

    <layer id="1" name="qsa" type="QwenSparseAttention" version="opset17">
        <data compress_ratio="4" token_budget="2048" index_element_type="i32"/>
        <input>
            <port id="0" precision="BF16"> <!-- query -->
                <dim>1</dim>   <!-- N -->
                <dim>24</dim>  <!-- H -->
                <dim>-1</dim>  <!-- L -->
                <dim>256</dim> <!-- E -->
            </port>
            <port id="1" precision="BF16"> <!-- key -->
                <dim>1</dim>   <!-- N -->
                <dim>2</dim>   <!-- Hkv -->
                <dim>-1</dim>  <!-- S -->
                <dim>256</dim> <!-- E -->
            </port>
            <port id="2" precision="BF16"> <!-- value -->
                <dim>1</dim>   <!-- N -->
                <dim>2</dim>   <!-- Hkv -->
                <dim>-1</dim>  <!-- S -->
                <dim>256</dim> <!-- Ev -->
            </port>
            <port id="3" precision="BF16"> <!-- index_query -->
                <dim>1</dim>   <!-- N -->
                <dim>4</dim>   <!-- Hi -->
                <dim>-1</dim>  <!-- L -->
                <dim>128</dim> <!-- Di -->
            </port>
            <port id="4" precision="BF16"> <!-- index_key_blocks -->
                <dim>1</dim>   <!-- N -->
                <dim>-1</dim>  <!-- B = floor(S / 4) -->
                <dim>128</dim> <!-- Di -->
            </port>
        </input>
        <output>
            <port id="5" precision="BF16"> <!-- output -->
                <dim>1</dim>   <!-- N -->
                <dim>24</dim>  <!-- H -->
                <dim>-1</dim>  <!-- L -->
                <dim>256</dim> <!-- Ev -->
            </port>
            <port id="6" precision="I32"> <!-- indices -->
                <dim>1</dim>    <!-- N -->
                <dim>-1</dim>   <!-- L -->
                <dim>2051</dim> <!-- token_budget + compress_ratio - 1 -->
            </port>
        </output>
    </layer>

*Example 2: One decode step over 10000 cached tokens with an explicit scale*

.. code-block:: xml
   :force:

    <layer id="2" name="qsa" type="QwenSparseAttention" version="opset17">
        <data compress_ratio="4" token_budget="2048"/>
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
            <port id="3" precision="FP32"> <!-- index_query -->
                <dim>2</dim>     <!-- N -->
                <dim>4</dim>     <!-- Hi -->
                <dim>1</dim>     <!-- L -->
                <dim>128</dim>   <!-- Di -->
            </port>
            <port id="4" precision="FP32"> <!-- index_key_blocks -->
                <dim>2</dim>     <!-- N -->
                <dim>2500</dim>  <!-- B = floor(10001 / 4) -->
                <dim>128</dim>   <!-- Di -->
            </port>
            <port id="5" precision="FP32"> <!-- scale -->
            </port>
        </input>
        <output>
            <!-- 512 of the 2500 blocks (2048 tokens) + 1 token of the incomplete block are attended -->
            <port id="6" precision="FP32"> <!-- output -->
                <dim>2</dim>     <!-- N -->
                <dim>24</dim>    <!-- H -->
                <dim>1</dim>     <!-- L -->
                <dim>256</dim>   <!-- Ev -->
            </port>
            <port id="7" precision="I32"> <!-- indices -->
                <dim>2</dim>     <!-- N -->
                <dim>1</dim>     <!-- L -->
                <dim>2051</dim>  <!-- token_budget + compress_ratio - 1 -->
            </port>
        </output>
    </layer>
