QSAPrepare
==========


.. meta::
  :description: Learn about QSAPrepare - builds the block index keys of Qwen Sparse Attention by
                pooling, normalizing and rotating the raw index keys of every complete key block.

**Versioned name**: *QSAPrepare*

**Category**: *Sequence processing*

**Short description**: *QSAPrepare* computes the block index keys of Qwen Sparse Attention (QSA):
one key per complete block of ``key_block_size`` consecutive key tokens, obtained by mean-pooling the
raw index keys of the block, RMS-normalizing the result and rotating it with the RoPE of the first
token of the block. The block keys of the past blocks are passed in from a cache, so that only the
blocks completed by the new tokens are computed.

**Detailed description**:

*QSAPrepare* is the first stage of a QSA layer in a model with
*ScaledDotProductAttention*-style caches. The graph flow is *QSAPrepare* ->
:doc:`SparseAttentionIndexer <sparse-attention-indexer>` -> :doc:`SparseAttention <sparse-attention>`; the paged
counterpart is :doc:`PagedQSAPrepare <paged-qsa-prepare>`. It covers the key compression of the
engines: the "compress K" part of the ``qsa_prepare`` kernel of vLLM and the pooling, normalization
and rotation of the block keys in ``build_qsa_top_k`` of llama.cpp.

In Qwen3.8-Flash-Next (``qwen4_exp`` in Hugging Face Transformers) one projection of the attention
layer input, ``index_qk_proj``, gives ``Hi = 4`` index query heads and one raw index key head, both of
size ``Di = 128``:

* the index query is prepared by regular operations of the graph, as the query of the main
  attention: every head RMS-normalized with ``q_layernorm`` (*RMS* with the weight ``1 + w``) and
  rotated with the (M-)RoPE of the query token (*RoPE*). It goes to *SparseAttentionIndexer* directly;
* the raw index keys go to *QSAPrepare*. A block key needs all ``key_block_size`` raw keys of its
  block and is rotated at the first token of the block, not at the position of each token, so it is
  not a per-token normalization and rotation: *QSAPrepare* mean-pools the raw keys of every complete
  block, RMS-normalizes the result with ``k_layernorm`` and rotates it with the (M-)RoPE of the first
  token of the block. The order pool -> normalize -> rotate is the order of the reference model.

A block key depends only on the raw keys of its own block and does not change when more tokens come.
The block keys computed by earlier calls are therefore cached: a stateful model keeps them as a state,
``past_block_index_keys`` ``[N, Bp, Di]``, next to the raw index keys. *QSAPrepare* computes only the
blocks completed since then, at most ``ceil(L / key_block_size) + 1`` blocks of a sequence for ``L``
new tokens, and returns all block keys, the cached ones followed by the new ones. Its output is the
new value of the state. A decode step completes at most one block, so the pooling, the normalization
and the rotation are done for at most two blocks per decode step instead of for all ``B`` blocks.

**Batches with padding.** The sequences of a batch are aligned to the right (left padding), as for
:doc:`SparseAttentionIndexer <sparse-attention-indexer>`: sequence ``n`` has ``key_length[n]`` valid key tokens, the last ones
of the ``S`` key positions. Its blocks start at its first valid token: block ``b`` is formed by the
key positions ``S - key_length[n] + b * key_block_size ..``, and ``block_index_keys[n, b]`` is the key
of that block. A padded sequence gets the same block keys as alone in a batch of one, and padding
tokens never enter a block key. The rows of a padded sequence after its last complete block are
zero.

*QSAPrepare* provides functionality according to the following pseudo-code using ``numpy``:

.. code-block:: py
   :force:

    def QSAPrepare(raw_index_keys, past_block_index_keys, key_length, rope_cos, rope_sin, k_norm_weight, *,
                   key_block_size, epsilon=1e-6):
        N, S, Di = raw_index_keys.shape
        r = key_block_size
        B = S // r                                              # blocks of a sequence without padding
        Bp = past_block_index_keys.shape[1]                     # cached block keys, Bp <= B
        key_length = numpy.broadcast_to(key_length, (N,))       # valid key tokens of every sequence
        Dr = rope_cos.shape[-1] if rope_cos.size else 0

        block_index_keys = numpy.zeros((N, B, Di), dtype=past_block_index_keys.dtype)
        for n in range(N):
            pad = S - key_length[n]                             # padding key positions, at the beginning
            num_blocks = key_length[n] // r                     # complete blocks of sequence n
            # the previous call had at least Bp * r key positions, so at least this many blocks of
            # sequence n are cached; they are reused, the others are computed. The cached rows of a
            # padded sequence after its valid blocks are fake and are not reused.
            num_cached = max(0, (Bp * r - pad) // r)
            block_index_keys[n, :num_cached] = past_block_index_keys[n, :num_cached]
            for b in range(num_cached, num_blocks):
                first = pad + b * r                             # key position of the first token of the block
                # 1. mean-pool the raw index keys of the block
                k = raw_index_keys[n, first:first + r].astype(numpy.float32).mean(axis=0)
                # 2. RMS-normalize
                if k_norm_weight.size:
                    k = k / numpy.sqrt((k * k).mean() + epsilon) * k_norm_weight
                # 3. rotate-half RoPE on the first Dr dims, at the first token of the block
                if Dr:
                    cos = rope_cos[min(n, rope_cos.shape[0] - 1), first]
                    sin = rope_sin[min(n, rope_sin.shape[0] - 1), first]
                    x = k[:Dr]
                    rotated = numpy.concatenate([-x[Dr // 2:], x[:Dr // 2]])
                    k = numpy.concatenate([x * cos + rotated * sin, k[Dr:]])
                block_index_keys[n, b] = k
            # the rows num_blocks .. B - 1 of a padded sequence stay zero: fake rows at the end
        return block_index_keys

Properties that follow from the definition:

* Block ``b`` of sequence ``n`` holds the key tokens at the positions
  ``S - key_length[n] + b * key_block_size ..`` (``b * key_block_size ..`` without padding) and is
  rotated with the ``rope_cos`` / ``rope_sin`` row of its first token: the rows of the other tokens of
  the block are not used. The tokens of the incomplete block, at most ``key_block_size - 1``, form no
  block key.
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
* The cached blocks of sequence ``n`` are reused unchanged. With an empty ``past_block_index_keys``
  (``Bp = 0``) all blocks are computed; when the block keys of the previous calls are passed in, the
  result is the same as without them, as a block key depends only on the raw keys of its own block.
  Without padding exactly the blocks ``Bp .. B - 1`` are computed. With padding, the previous call had
  between ``Bp * key_block_size`` and ``Bp * key_block_size + key_block_size - 1`` key positions, so
  the number of cached blocks of a padded sequence is known up to one: the operation may recompute
  one cached block, with the same result.
* Only the raw keys and the ``rope_cos`` / ``rope_sin`` rows of the computed blocks are read. The first
  ones may be past tokens of the incomplete block of the previous call.
* For DeepSeek Sparse Attention (``key_block_size = 1``), pooling over one token is the identity and
  the index key is a LayerNorm and a partial RoPE per token, computed by regular operations of the
  graph; *QSAPrepare* is not needed.

.. note::

   **Padding layout: which entries are fake.** With ``pad = S - key_length[n]`` padding tokens of
   sequence ``n``:

   * ``raw_index_keys``, ``rope_cos`` and ``rope_sin`` are padded at the **beginning**: the key
     positions ``0 .. pad - 1`` are padding and never enter a block key.
   * ``block_index_keys`` and ``past_block_index_keys`` are **left-aligned**: the blocks of a
     sequence are counted from its first valid token, so the valid rows come first and the fake rows
     are at the **end**. In the output, the rows ``floor(key_length[n] / key_block_size) ..`` of a
     padded sequence are fake and zero. In ``past_block_index_keys``, the rows
     ``floor(key_length_previous[n] / key_block_size) ..`` are fake, where ``key_length_previous`` is the
     ``key_length`` of the previous call; the operation reuses only the first
     ``max(0, (Bp * key_block_size - pad) // key_block_size)`` rows, a lower bound that needs no
     ``key_length_previous``, and recomputes at most one valid cached block.

   See the note *Padding layout* of :doc:`SparseAttentionIndexer <sparse-attention-indexer>` for the query axis and an
   example:

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


**Attributes**

* *key_block_size*

  * **Description**: number of consecutive key tokens that form one block. It must be equal to
    *key_block_size* of the consuming *SparseAttentionIndexer*.
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

* **2**: ``past_block_index_keys`` - 3D tensor of type *T* and shape ``[N, Bp, Di]``: the block index
  keys of the first ``Bp`` complete blocks, computed by the previous calls of the same sequences,
  ``Bp <= floor(S / key_block_size)``. In a stateful model it is a state, and the output is its new
  value. **Required (empty tensor with Bp = 0 = no cached blocks, all blocks are computed).**

* **3**: ``key_length`` - 1D tensor of type *T_IND* and shape ``[N]``: the number of valid key tokens
  of every sequence, ``key_length[n] <= S``; the same input as ``key_length`` of *SparseAttentionIndexer*. A tensor
  of shape ``[1]`` applies to all sequences; without padding it is ``S``. **Required.**

* **4**: ``rope_cos`` - 3D tensor of type *T* and shape ``[N, S, Dr]``, or an empty tensor: the RoPE
  ``cos`` of every key position, numpy-broadcastable over ``N``. Only the rows of the first tokens of
  the computed blocks are used. **Required (empty tensor = no rotation).**

* **5**: ``rope_sin`` - 3D tensor of type *T* and shape ``[N, S, Dr]``, or an empty tensor: the RoPE
  ``sin`` of every key token. **Required (empty tensor = no rotation).**

* **6**: ``k_norm_weight`` - 1D tensor of type *T* and shape ``[Di]``, or an empty tensor: the scale
  of the block key normalization. **Required (empty tensor = no normalization).**


**Outputs**

* **1**: ``block_index_keys`` - 3D tensor of type *T* and shape ``[N, B, Di]``: one normalized and
  rotated key per complete key block of every sequence: the cached block keys followed by the keys
  of the new blocks, zero after the last complete block of a padded sequence. Input
  ``block_index_keys`` of *SparseAttentionIndexer* and the new value of the block key cache.


**Types**

* *T*: any supported floating-point type.

* *T_IND*: ``int32`` or ``int64``.


**Dimensions**

* ``N`` - batch size. The sequences of a batch are aligned to the right (left padding), see
  ``key_length``.

* ``S`` - number of key positions, including the query tokens and the padding.

* ``Di`` - head size of the index key.

* ``Dr`` - number of rotated dimensions, even, ``Dr <= Di``.

* ``B`` - ``floor(S / key_block_size)``, the number of complete blocks of a sequence without padding.
  A padded sequence has ``floor(key_length[n] / key_block_size)``.

* ``Bp`` - number of cached block keys, ``Bp <= B``.


**Example**

*Qwen3.8-Flash-Next, one decode step over 10003 cached tokens: the new token completes block 2500*

.. code-block:: xml
   :force:

    <layer id="1" name="qsa_prepare" type="QSAPrepare" version="extension">
        <data key_block_size="4" epsilon="1e-06"/>
        <input>
            <port id="0" precision="BF16"> <!-- raw_index_keys, past state + current -->
                <dim>1</dim>     <!-- N -->
                <dim>10004</dim> <!-- S -->
                <dim>128</dim>   <!-- Di -->
            </port>
            <port id="1" precision="BF16"> <!-- past_block_index_keys, state -->
                <dim>1</dim>     <!-- N -->
                <dim>2500</dim>  <!-- Bp -->
                <dim>128</dim>   <!-- Di -->
            </port>
            <port id="2" precision="I64"> <dim>1</dim> </port> <!-- key_length = [10004] -->
            <port id="3" precision="BF16"> <!-- rope_cos of the main attention -->
                <dim>1</dim>     <!-- N -->
                <dim>10004</dim> <!-- S -->
                <dim>64</dim>    <!-- Dr -->
            </port>
            <port id="4" precision="BF16"> <!-- rope_sin -->
                <dim>1</dim>
                <dim>10004</dim>
                <dim>64</dim>
            </port>
            <port id="5" precision="BF16"> <dim>128</dim> </port> <!-- k_norm_weight = 1 + k_layernorm.weight -->
        </input>
        <output>
            <port id="6" precision="BF16"> <!-- block_index_keys: 2500 cached + 1 new, new state value -->
                <dim>1</dim>     <!-- N -->
                <dim>2501</dim>  <!-- B = floor(10004 / 4) -->
                <dim>128</dim>   <!-- Di -->
            </port>
        </output>
    </layer>
