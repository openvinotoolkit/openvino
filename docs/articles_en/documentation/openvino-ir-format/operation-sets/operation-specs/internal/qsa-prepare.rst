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
token of the block. It takes the raw index keys and the RoPE of the new tokens only: the block keys
of the past blocks, and the raw keys and the RoPE row of the past tokens of the incomplete block, are
passed in from small caches, so that only the blocks completed by the new tokens are computed.

**Detailed description**:

*QSAPrepare* is the first stage of a QSA layer in a model with
*ScaledDotProductAttention*-style caches. The graph flow is *QSAPrepare* ->
:doc:`SparseAttentionIndexer <../sequence/sparse-attention-indexer-17>` -> :doc:`SparseAttention <../sequence/sparse-attention-17>`; the paged
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
``past_block_index_keys`` ``[N, Bp, Di]``. The raw keys of a complete block are never read again, so
the raw index keys of the past tokens are not cached: only the tokens of the incomplete block, at most
``key_block_size - 1`` of them, are needed by a later call, together with the RoPE row of the first
token of that block, at which the block key is rotated. A stateful model keeps them in two fixed-size
states, ``past_incomplete_raw_keys`` ``[N, key_block_size - 1, Di]`` and ``past_incomplete_rope``
``[N, 2 * Dr]``. *QSAPrepare* takes the raw index keys and the ``rope_cos`` / ``rope_sin`` of the ``L``
new tokens, computes the blocks completed by them, at most ``ceil(L / key_block_size)`` blocks of a
sequence, and returns all block keys, the cached ones followed by the new ones, together with the new
values of the two incomplete-block states. A decode step completes at most one block, so the pooling,
the normalization and the rotation are done for at most one block per decode step, and no input or
state grows with the number of past tokens except the block keys.

**Batches with padding.** The sequences of a batch are aligned to the right (left padding), as for
:doc:`SparseAttentionIndexer <../sequence/sparse-attention-indexer-17>`: sequence ``n`` has ``key_length[n]`` valid key tokens, past
and current, the last ones of all its key positions. Its blocks start at its first valid token:
block ``b`` is formed by its valid tokens ``b * key_block_size .. b * key_block_size + key_block_size - 1``,
and ``block_index_keys[n, b]`` is the key of that block. Of its ``key_length[n]`` valid tokens,
``past_n = max(0, key_length[n] - L)`` come from earlier calls and the others are the last rows of the
``L`` new tokens; only a pre-fill can contain padding rows, at the beginning. A padded sequence gets
the same block keys as alone in a batch of one, and padding tokens never enter a block key. The rows
of a padded sequence after its last complete block are zero.

*QSAPrepare* provides functionality according to the following pseudo-code using ``numpy``:

.. code-block:: py
   :force:

    def QSAPrepare(raw_index_keys, past_block_index_keys, key_length, rope_cos, rope_sin, k_norm_weight,
                   past_incomplete_raw_keys, past_incomplete_rope, *, key_block_size, epsilon=1e-6):
        N, L, Di = raw_index_keys.shape
        r = key_block_size
        key_length = numpy.broadcast_to(key_length, (N,))       # valid key tokens of every sequence, past and current
        Dr = rope_cos.shape[-1] if rope_cos.size else 0
        B = max(int(k) // r for k in key_length)                # complete blocks of the longest sequence

        block_index_keys = numpy.zeros((N, B, Di), dtype=past_block_index_keys.dtype)
        incomplete_raw_keys = numpy.zeros((N, r - 1, Di), dtype=raw_index_keys.dtype)
        incomplete_rope = numpy.zeros((N, 2 * Dr), dtype=rope_cos.dtype)
        for n in range(N):
            past = max(0, int(key_length[n]) - L)               # valid tokens of the earlier calls
            current = int(key_length[n]) - past                 # valid new tokens, the last rows of the L ones
            c = past % r                                        # past tokens of the incomplete block
            num_cached = past // r                              # complete blocks of the earlier calls
            block_index_keys[n, :num_cached] = past_block_index_keys[n, :num_cached]
            # the tokens not yet in a block key: the incomplete block of the earlier calls, then the new tokens
            keys = numpy.concatenate([past_incomplete_raw_keys[n, :c], raw_index_keys[n, L - current:]])

            def rope_row(t):                                    # cos, sin of the pending token t
                if t < c:                                       # a past token: the first one of the incomplete block
                    return past_incomplete_rope[n, :Dr], past_incomplete_rope[n, Dr:]
                row = L - current + t - c
                return rope_cos[min(n, rope_cos.shape[0] - 1), row], rope_sin[min(n, rope_sin.shape[0] - 1), row]

            num_new = len(keys) // r                            # blocks completed by the new tokens
            for j in range(num_new):
                # 1. mean-pool the raw index keys of the block
                k = keys[j * r:(j + 1) * r].astype(numpy.float32).mean(axis=0)
                # 2. RMS-normalize
                if k_norm_weight.size:
                    k = k / numpy.sqrt((k * k).mean() + epsilon) * k_norm_weight
                # 3. rotate-half RoPE on the first Dr dims, at the first token of the block
                if Dr:
                    cos, sin = rope_row(j * r)
                    x = k[:Dr]
                    rotated = numpy.concatenate([-x[Dr // 2:], x[:Dr // 2]])
                    k = numpy.concatenate([x * cos + rotated * sin, k[Dr:]])
                block_index_keys[n, num_cached + j] = k
            # the rows num_cached + num_new .. B - 1 of a padded sequence stay zero: fake rows at the end
            # 4. the incomplete block after this call, for the next call
            rest = len(keys) - num_new * r
            incomplete_raw_keys[n, :rest] = keys[num_new * r:]
            if rest and Dr:
                incomplete_rope[n] = numpy.concatenate(rope_row(num_new * r))
        return block_index_keys, incomplete_raw_keys, incomplete_rope

Properties that follow from the definition:

* Block ``b`` of sequence ``n`` holds its valid key tokens ``b * key_block_size ..`` and is rotated
  with the ``rope_cos`` / ``rope_sin`` row of its first token: the rows of the other tokens of the
  block are not used. The tokens of the incomplete block, at most ``key_block_size - 1``, form no
  block key; they are returned in ``incomplete_raw_keys``, with the RoPE row of their first token in
  ``incomplete_rope``.
* The result does not depend on how the tokens of a sequence are split into calls: with the states of
  the previous call passed in, the block keys are the ones of the whole sequence, computed from the
  raw index keys and the RoPE rows of all its tokens. A new sequence (``past_n = 0``) does not read
  the states, so their initial content does not matter.
* In a call, only the first block completed by sequence ``n`` can contain past tokens: the
  ``c = past_n mod key_block_size`` rows of ``past_incomplete_raw_keys``, rotated with
  ``past_incomplete_rope``. The cos / sin of a past token is persisted rather than recomputed, which
  also covers M-RoPE positions of image and video tokens that cannot be derived from a token count.
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
* The first ``floor(past_n / key_block_size)`` blocks of sequence ``n`` are taken from
  ``past_block_index_keys`` unchanged and exactly the blocks completed since then are computed. The
  inputs and the states have a size independent of the number of past tokens, except
  ``past_block_index_keys``, which has one row per complete block.
* For DeepSeek Sparse Attention (``key_block_size = 1``), pooling over one token is the identity and
  the index key is a LayerNorm and a partial RoPE per token, computed by regular operations of the
  graph; *QSAPrepare* is not needed and requires ``key_block_size >= 2``.

.. note::

   **Padding layout: which entries are fake.** Sequence ``n`` has ``past_n = max(0, key_length[n] - L)``
   valid past tokens and ``key_length[n] - past_n`` valid new tokens:

   * ``raw_index_keys``, ``rope_cos`` and ``rope_sin`` (the ``L`` new tokens) are padded at the
     **beginning**: only a pre-fill can have padding rows, ``0 .. L - key_length[n] - 1``, which never
     enter a block key.
   * ``block_index_keys`` and ``past_block_index_keys`` are **left-aligned**: the blocks of a
     sequence are counted from its first valid token, so the valid rows come first and the fake rows
     are at the **end**. In the output, the rows ``floor(key_length[n] / key_block_size) ..`` of a
     padded sequence are fake and zero. In ``past_block_index_keys`` the first
     ``floor(past_n / key_block_size)`` rows are valid and reused.
   * ``past_incomplete_raw_keys`` and ``incomplete_raw_keys`` are **left-aligned**: the first
     ``past_n mod key_block_size`` (``key_length[n] mod key_block_size`` in the output) rows are the raw
     keys of the tokens of the incomplete block, oldest first; the other rows are fake and zero in the
     output. ``past_incomplete_rope`` and ``incomplete_rope`` are valid only if that number is not zero.

   See the note *Padding layout* of :doc:`SparseAttentionIndexer <../sequence/sparse-attention-indexer-17>` for the query axis and an
   example:

   .. code-block:: text
      :force:

      key_block_size = 4, pre-fill: L = 10, key_length = [10, 7]

      new token row:  0   1   2   3   4   5   6   7   8   9
      sequence 0:     a0  a1  a2  a3  a4  a5  a6  a7  a8  a9     past = 0
      sequence 1:     .   .   .   c0  c1  c2  c3  c4  c5  c6     past = 0   (. = padding)

      block_index_keys[0]    = [ key(a0..a3), key(a4..a7) ]
      block_index_keys[1]    = [ key(c0..c3), 0 (fake)    ]
      incomplete_raw_keys[0] = [ a8, a9, 0 ],     incomplete_rope[0] = cos, sin of a8
      incomplete_raw_keys[1] = [ c4, c5, c6 ],    incomplete_rope[1] = cos, sin of c4

      next decode step (L = 1, key_length = [11, 8]): past = [10, 7], c = [2, 3]
      sequence 0: a10 joins a8 a9, no block is completed;   incomplete_raw_keys[0] = [ a8, a9, a10 ]
      sequence 1: c7 completes the block c4..c7, block 1, rotated with the cos, sin of c4 from
                  past_incomplete_rope;                       incomplete_raw_keys[1] = [ 0, 0, 0 ]


**Attributes**

* *key_block_size*

  * **Description**: number of consecutive key tokens that form one block. It must be equal to
    *key_block_size* of the consuming *SparseAttentionIndexer*. Single-token blocks
    (``key_block_size = 1``, DeepSeek Sparse Attention) need no pooling and no *QSAPrepare*, see the
    properties above.
  * **Range of values**: an integer ``>= 2``
  * **Type**: ``int``
  * **Required**: *yes*

* *epsilon*

  * **Description**: the value added to the mean square in the RMS normalization.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: ``1e-6``
  * **Required**: *no*


**Inputs**

* **1**: ``raw_index_keys`` - 3D tensor of type *T* and shape ``[N, L, Di]``: the raw index key of
  every new token, before normalization and rotation. **Required.**

* **2**: ``past_block_index_keys`` - 3D tensor of type *T* and shape ``[N, Bp, Di]``: the block index
  keys of the complete blocks computed by the previous calls of the same sequences: the first
  ``floor(past_n / key_block_size)`` rows of sequence ``n`` are valid. In a stateful model it is a
  state, and output 1 is its new value. **Required (empty tensor with Bp = 0 for new sequences).**

* **3**: ``key_length`` - 1D tensor of type *T_IND* and shape ``[N]``: the number of valid key tokens
  of every sequence, past and current; the same input as ``key_length`` of *SparseAttentionIndexer*. A
  tensor of shape ``[1]`` applies to all sequences. **Required.**

* **4**: ``rope_cos`` - 3D tensor of type *T* and shape ``[N, L, Dr]``, or an empty tensor: the RoPE
  ``cos`` of every new token, numpy-broadcastable over ``N``. Only the rows of the first tokens of
  the computed blocks and of the incomplete block are used. **Required (empty tensor = no rotation).**

* **5**: ``rope_sin`` - 3D tensor of type *T* and shape ``[N, L, Dr]``, or an empty tensor: the RoPE
  ``sin`` of every new token. **Required (empty tensor = no rotation).**

* **6**: ``k_norm_weight`` - 1D tensor of type *T* and shape ``[Di]``, or an empty tensor: the scale
  of the block key normalization. **Required (empty tensor = no normalization).**

* **7**: ``past_incomplete_raw_keys`` - 3D tensor of type *T* and shape ``[N, key_block_size - 1, Di]``:
  the raw index keys of the past tokens of the incomplete block of every sequence, oldest first; the
  first ``past_n mod key_block_size`` rows are valid. In a stateful model it is a state, and output 2 is
  its new value. **Required.**

* **8**: ``past_incomplete_rope`` - 2D tensor of type *T* and shape ``[N, 2 * Dr]``, or an empty tensor:
  ``rope_cos`` followed by ``rope_sin`` of the first token of the incomplete block of every sequence,
  valid if ``past_n mod key_block_size`` is not zero. In a stateful model it is a state, and output 3 is
  its new value. **Required (empty tensor if and only if rope_cos and rope_sin are empty).**


**Outputs**

* **1**: ``block_index_keys`` - 3D tensor of type *T* and shape ``[N, B, Di]``: one normalized and
  rotated key per complete key block of every sequence: the cached block keys followed by the keys
  of the new blocks, zero after the last complete block of a padded sequence. Input
  ``block_index_keys`` of *SparseAttentionIndexer* and the new value of ``past_block_index_keys``.

* **2**: ``incomplete_raw_keys`` - 3D tensor of type *T* and shape ``[N, key_block_size - 1, Di]``: the
  raw index keys of the tokens of the incomplete block after this call, oldest first, the first
  ``key_length[n] mod key_block_size`` rows; the other rows are zero. The new value of
  ``past_incomplete_raw_keys``.

* **3**: ``incomplete_rope`` - 2D tensor of type *T* and shape ``[N, 2 * Dr]``, or an empty tensor: the
  ``cos`` and ``sin`` of the first token of the incomplete block after this call, zero if
  ``key_length[n] mod key_block_size`` is zero. The new value of ``past_incomplete_rope``.


**Types**

* *T*: any supported floating-point type.

* *T_IND*: ``int32`` or ``int64``.


**Dimensions**

* ``N`` - batch size. The sequences of a batch are aligned to the right (left padding), see
  ``key_length``.

* ``L`` - number of new tokens, including the padding of a pre-fill.

* ``Di`` - head size of the index key.

* ``Dr`` - number of rotated dimensions, even, ``Dr <= Di``.

* ``B`` - ``max(floor(key_length[n] / key_block_size))``, the number of complete blocks of the longest
  sequence. A padded sequence has ``floor(key_length[n] / key_block_size)``.

* ``Bp`` - number of cached block keys, ``B`` of the previous call.

* ``past_n`` - ``max(0, key_length[n] - L)``, the number of valid tokens of sequence ``n`` processed by
  the previous calls.


**Example**

*Qwen3.8-Flash-Next, one decode step over 10003 cached tokens: the new token completes block 2500*

The previous calls processed 10003 tokens: 2500 complete blocks and the tokens 10000 .. 10002 of the
incomplete block. The new token 10003 completes block 2500, rotated with the cos / sin of token 10000
kept in ``past_incomplete_rope``.

.. code-block:: xml
   :force:

    <layer id="1" name="qsa_prepare" type="QSAPrepare" version="extension">
        <data key_block_size="4" epsilon="1e-06"/>
        <input>
            <port id="0" precision="BF16"> <!-- raw_index_keys of the new token -->
                <dim>1</dim>     <!-- N -->
                <dim>1</dim>     <!-- L -->
                <dim>128</dim>   <!-- Di -->
            </port>
            <port id="1" precision="BF16"> <!-- past_block_index_keys, state -->
                <dim>1</dim>     <!-- N -->
                <dim>2500</dim>  <!-- Bp -->
                <dim>128</dim>   <!-- Di -->
            </port>
            <port id="2" precision="I64"> <dim>1</dim> </port> <!-- key_length = [10004] -->
            <port id="3" precision="BF16"> <!-- rope_cos of the new token, from the main attention -->
                <dim>1</dim>     <!-- N -->
                <dim>1</dim>     <!-- L -->
                <dim>64</dim>    <!-- Dr -->
            </port>
            <port id="4" precision="BF16"> <!-- rope_sin -->
                <dim>1</dim>
                <dim>1</dim>
                <dim>64</dim>
            </port>
            <port id="5" precision="BF16"> <dim>128</dim> </port> <!-- k_norm_weight = 1 + k_layernorm.weight -->
            <port id="6" precision="BF16"> <!-- past_incomplete_raw_keys, state: tokens 10000 .. 10002 -->
                <dim>1</dim>     <!-- N -->
                <dim>3</dim>     <!-- key_block_size - 1 -->
                <dim>128</dim>   <!-- Di -->
            </port>
            <port id="7" precision="BF16"> <!-- past_incomplete_rope, state: cos, sin of token 10000 -->
                <dim>1</dim>     <!-- N -->
                <dim>128</dim>   <!-- 2 * Dr -->
            </port>
        </input>
        <output>
            <port id="8" precision="BF16"> <!-- block_index_keys: 2500 cached + 1 new, new state value -->
                <dim>1</dim>     <!-- N -->
                <dim>2501</dim>  <!-- B = floor(10004 / 4) -->
                <dim>128</dim>   <!-- Di -->
            </port>
            <port id="9" precision="BF16"> <!-- incomplete_raw_keys: empty block, zeros -->
                <dim>1</dim>
                <dim>3</dim>
                <dim>128</dim>
            </port>
            <port id="10" precision="BF16"> <!-- incomplete_rope: zeros -->
                <dim>1</dim>
                <dim>128</dim>
            </port>
        </output>
    </layer>
