QSAPrepare
==========


.. meta::
  :description: Learn about QSAPrepare - builds the block index keys of Qwen Sparse Attention by
                pooling, normalizing and rotating the raw index keys of every complete key block.

**Versioned name**: *QSAPrepare*

**Category**: *Sequence processing*

**Short description**: *QSAPrepare* computes the block index keys of Qwen Sparse Attention (QSA):
one key per complete block of ``compress_ratio`` consecutive key tokens, obtained by mean-pooling the
raw index keys of the block, RMS-normalizing the result and rotating it with the RoPE of the first
token of the block.

**Detailed description**:

*QSAPrepare* is the first stage of a QSA layer in a model with
*ScaledDotProductAttention*-style caches. The graph flow is *QSAPrepare* ->
:doc:`QSAIndexer <qsa-indexer>` -> :doc:`BlockSparseAttention <block-sparse-attention>`; the paged
counterpart is :doc:`PagedQSAPrepare <paged-qsa-prepare>`. It covers the key compression of the
engines: the "compress K" part of the ``qsa_prepare`` kernel of vLLM and the pooling, normalization
and rotation of the block keys in ``build_qsa_top_k`` of llama.cpp.

In Qwen3.8-Flash-Next (``qwen4_exp`` in Hugging Face Transformers) one projection of the attention
layer input, ``index_qk_proj``, gives ``Hi = 4`` index query heads and one raw index key head, both of
size ``Di = 128``:

* the index query is prepared by regular operations of the graph, as the query of the main
  attention: every head RMS-normalized with ``q_layernorm`` (*RMS* with the weight ``1 + w``) and
  rotated with the (M-)RoPE of the query token (*RoPE*). It goes to *QSAIndexer* directly;
* the raw index keys go to *QSAPrepare*. A block key needs all ``compress_ratio`` raw keys of its
  block and is rotated at the first token of the block, not at the position of each token, so it is
  not a per-token normalization and rotation: *QSAPrepare* mean-pools the raw keys of every complete
  block, RMS-normalizes the result with ``k_layernorm`` and rotates it with the (M-)RoPE of the first
  token of the block. The order pool -> normalize -> rotate is the order of the reference model.

*QSAPrepare* provides functionality according to the following pseudo-code using ``numpy``:

.. code-block:: py
   :force:

    def QSAPrepare(raw_index_keys, rope_cos, rope_sin, k_norm_weight, *, compress_ratio, epsilon=1e-6):
        N, S, Di = raw_index_keys.shape
        r = compress_ratio
        B = S // r                                              # complete key blocks
        Dr = rope_cos.shape[-1] if rope_cos.size else 0

        # 1. mean-pool the raw index keys of every complete block
        k = raw_index_keys[:, :B * r].reshape(N, B, r, Di).astype(numpy.float32).mean(axis=2)
        # 2. RMS-normalize every block key
        if k_norm_weight.size:
            k = k / numpy.sqrt((k * k).mean(axis=-1, keepdims=True) + epsilon) * k_norm_weight
        # 3. rotate-half RoPE on the first Dr dims, at the position of the first token of every block
        if Dr:
            starts = numpy.arange(B) * r
            cos, sin = rope_cos[:, starts, :], rope_sin[:, starts, :]
            x = k[..., :Dr]
            rotated = numpy.concatenate([-x[..., Dr // 2:], x[..., :Dr // 2]], axis=-1)
            k = numpy.concatenate([x * cos + rotated * sin, k[..., Dr:]], axis=-1)
        return k                                                # block_index_keys [N, B, Di]

Properties that follow from the definition:

* Block ``b`` holds the key tokens ``b * compress_ratio .. b * compress_ratio + compress_ratio - 1``
  and is rotated with the row ``b * compress_ratio`` of ``rope_cos`` and ``rope_sin``: the rows of
  the other tokens of the block are not used. The tokens of the incomplete block, at most
  ``compress_ratio - 1``, form no block key.
* The normalization is ``x / sqrt(mean(x^2) + epsilon) * weight``. Qwen3.8-Flash-Next normalizes with
  ``k_layernorm``, which scales by ``1 + w``; the graph passes ``1 + w`` as ``k_norm_weight``. An empty
  ``k_norm_weight`` disables the normalization.
* The rotation is the split-half (rotate-half) RoPE of Hugging Face Transformers on the first ``Dr``
  dimensions, ``x * cos + rotate_half(x) * sin`` with ``Dr`` the last dimension of ``rope_cos``; the
  dimensions after ``Dr`` are not rotated. The ``cos`` / ``sin`` are the ones of the main attention,
  with M-RoPE multi-axis positions already resolved by the graph, so the operation does not depend on
  ``rope_theta``, the axis interleaving or a scaling method. For Qwen3.8-Flash-Next ``Dr = 64`` of
  ``Di = 128``. Empty ``rope_cos`` and ``rope_sin`` disable the rotation.
* The pooling and the normalization are computed with at least ``f32`` precision. The reference model
  rounds to its element type after the pooling and after the normalization; an implementation that
  must match a ``bf16`` reference bit by bit rounds at the same points.
* A block key depends only on the raw keys of its own block. A stateful model may keep the block
  index keys of the past blocks as a state and compute only the blocks completed by the new tokens;
  the result is the same.
* For DeepSeek Sparse Attention (``compress_ratio = 1``), pooling over one token is the identity and
  the index key is a LayerNorm and a partial RoPE per token, computed by regular operations of the
  graph; *QSAPrepare* is not needed.


**Attributes**

* *compress_ratio*

  * **Description**: number of consecutive key tokens that form one block. It must be equal to
    *compress_ratio* of the consuming *QSAIndexer*.
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

* **1**: ``raw_index_keys`` - 3D tensor of type *T* and shape ``[N, S, Di]``: the raw index key of
  every key token, past and current, before normalization and rotation. **Required.**

* **2**: ``rope_cos`` - 3D tensor of type *T* and shape ``[N, S, Dr]``, or an empty tensor: the RoPE
  ``cos`` of every key token, numpy-broadcastable over ``N``. Only the rows of the block starts are
  used. **Required (empty tensor = no rotation).**

* **3**: ``rope_sin`` - 3D tensor of type *T* and shape ``[N, S, Dr]``, or an empty tensor: the RoPE
  ``sin`` of every key token. **Required (empty tensor = no rotation).**

* **4**: ``k_norm_weight`` - 1D tensor of type *T* and shape ``[Di]``, or an empty tensor: the scale
  of the block key normalization. **Required (empty tensor = no normalization).**


**Outputs**

* **1**: ``block_index_keys`` - 3D tensor of type *T* and shape ``[N, B, Di]``: one normalized and
  rotated key per complete key block, input ``block_index_keys`` of *QSAIndexer*.


**Types**

* *T*: any supported floating-point type.


**Dimensions**

* ``N`` - batch size. The sequences of a batch are aligned to the right. Batches with padding are not
  supported.

* ``S`` - number of key tokens, including the query tokens.

* ``Di`` - head size of the index key.

* ``Dr`` - number of rotated dimensions, even, ``Dr <= Di``.

* ``B`` - number of complete key blocks, ``floor(S / compress_ratio)``.


**Example**

*Qwen3.8-Flash-Next, one decode step over 10000 cached tokens*

.. code-block:: xml
   :force:

    <layer id="1" name="qsa_prepare" type="QSAPrepare" version="extension">
        <data compress_ratio="4" epsilon="1e-06"/>
        <input>
            <port id="0" precision="BF16"> <!-- raw_index_keys, past state + current -->
                <dim>1</dim>     <!-- N -->
                <dim>10001</dim> <!-- S -->
                <dim>128</dim>   <!-- Di -->
            </port>
            <port id="1" precision="BF16"> <!-- rope_cos of the main attention -->
                <dim>1</dim>     <!-- N -->
                <dim>10001</dim> <!-- S -->
                <dim>64</dim>    <!-- Dr -->
            </port>
            <port id="2" precision="BF16"> <!-- rope_sin -->
                <dim>1</dim>
                <dim>10001</dim>
                <dim>64</dim>
            </port>
            <port id="3" precision="BF16"> <dim>128</dim> </port> <!-- k_norm_weight = 1 + k_layernorm.weight -->
        </input>
        <output>
            <port id="4" precision="BF16"> <!-- block_index_keys -->
                <dim>1</dim>     <!-- N -->
                <dim>2500</dim>  <!-- B = floor(10001 / 4) -->
                <dim>128</dim>   <!-- Di -->
            </port>
        </output>
    </layer>
