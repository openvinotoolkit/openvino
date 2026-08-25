BlockSparseAttention
====================


.. meta::
  :description: Learn about BlockSparseAttention-17 - a fused block-sparse scaled dot-product
                attention operation.

**Versioned name**: *BlockSparseAttention-17*

**Category**: *Sequence processing*

**Short description**: *BlockSparseAttention* computes exact scaled dot-product attention in
which every query block only attends to a caller-supplied, per-query-block selection of
key/value blocks, fusing the "gather selected blocks" step with the attention computation
itself.

**Detailed description**:

Query, key and value tokens are conceptually partitioned into contiguous, non-overlapping
blocks of ``block_size`` tokens along the sequence dimension. For every query block, the
``block_indices`` input names which key/value blocks that query block is allowed to attend
to; only tokens belonging to those blocks contribute to the attention output, all other key/
value tokens are excluded (not merely down-weighted). ``block_indices`` is typically produced
upstream (for example by pooling query/key blocks and running TopK on the resulting coarse
scores) — this operation only fuses the "gather the selected blocks" and "attend" steps, it
does not compute the selection itself.

*BlockSparseAttention* provides functionality according to the following pseudo-code, using
other operations from the OpenVINO opset and ``numpy``:

.. code-block:: py
    :force:

    def BlockSparseAttention(query, key, value, block_indices, block_indices_mask=None, scale=None, *, block_size, causal):
        B, H, L, E = query.shape
        _, Hk, S, Ev = value.shape
        _, Hb, num_q_blocks, k_blocks = block_indices.shape
        num_kv_blocks = S // block_size
        if scale is None:
            scale = 1.0 / np.sqrt(E)

        output = np.zeros((B, H, L, Ev), dtype=query.dtype)
        for b, h in np.ndindex(B, H):
            hk = 0 if Hk == 1 else h  # key/value may broadcast a single shared head
            hb = 0 if Hb == 1 else h  # block_indices/mask may broadcast a single shared head
            for qb in range(num_q_blocks):
                # Gather this query block's selected key/value blocks -- happens *inside*
                # the operation, no materialized gathered/transposed copy is required.
                selected = [blk for i, blk in enumerate(block_indices[b, hb, qb])
                            if block_indices_mask is None or block_indices_mask[b, hb, qb, i]]
                key_positions = [blk * block_size + t for blk in selected for t in range(block_size)
                                  if 0 <= blk < num_kv_blocks]
                for qi in range(block_size):
                    q_pos = qb * block_size + qi
                    positions = [p for p in key_positions if not causal or p <= q_pos]
                    logits = [query[b, h, q_pos] @ key[b, hk, p] * scale for p in positions]
                    weights = softmax(logits)  # empty input -> all-zero output row
                    output[b, h, q_pos] = sum(w * value[b, hk, p] for w, p in zip(weights, positions))
        return output

Unlike an equivalent subgraph built from existing operations (``Gather`` + ``Reshape`` +
``ScaledDotProductAttention``), *BlockSparseAttention* never materializes a gathered and
transposed copy of ``key``/``value``: it reads directly from the original tensors at the
offsets named by ``block_indices``. The cost of the decomposed subgraph scales with
``batch * heads * num_query_blocks * gathered_length`` regardless of how sparse the
selection actually is, while *BlockSparseAttention* scales with the amount of selected
(sparse) work only.

A ``block_indices`` entry that falls outside ``[0, num_kv_blocks)`` is treated the same as a
masked-out entry (ignored, contributes nothing to the output), rather than being an error --
this allows a fixed-size ``block_indices`` tensor to be reused across queries with a
different number of actually-relevant candidate blocks, by padding unused slots with an
out-of-range sentinel index in addition to (or instead of) the explicit
``block_indices_mask``. When ``causal`` is ``true``, this same "contributes nothing" handling
also applies token-by-token to any selected key position that lies after the query position,
including a key/value block that only partially precedes the query block on the diagonal, so
callers are not required to pre-filter ``block_indices`` for causal validity.

**Attributes**

* *block_size*

  * **Description**: number of contiguous key/value tokens per block. The query sequence
    length and the key/value sequence length must each be evenly divisible by *block_size*.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *causal*

  * **Description**: if ``true``, in addition to the block-level selection expressed by
    ``block_indices``, applies token-level causal masking so that query position ``i`` never
    attends to a key position greater than ``i``, even if that key position belongs to a
    selected block.
  * **Range of values**: a boolean value
  * **Type**: ``bool``
  * **Default value**: ``false``
  * **Required**: *no*

**Inputs**

* **1**: ``query`` - 4-dimensional tensor of type *T* and shape ``[B, H, L, E]``.
  **Required.**

* **2**: ``key`` - 4-dimensional tensor of type *T* and shape ``[B, Hk, S, E]``, where ``Hk``
  is either equal to ``H`` (one key head per query head) or ``1`` (a single key head
  broadcast to every query head, mirroring the level of head broadcasting supported by
  *ScaledDotProductAttention*). **Required.**

* **3**: ``value`` - 4-dimensional tensor of type *T* and shape ``[B, Hk, S, Ev]``, using the
  same ``Hk`` as ``key``. **Required.**

* **4**: ``block_indices`` - 4-dimensional tensor of type *T_IDX* and shape
  ``[B, Hb, L / block_size, k_blocks]``, containing, for every query block, the indices (into
  the ``S / block_size`` key/value blocks) of the key/value blocks selected for that query
  block. ``Hb`` is either ``H`` or ``1`` (broadcast to every query head), the same rule as
  ``Hk`` above. An index outside ``[0, S / block_size)`` is treated as masked out (see
  *Detailed description*). **Required.**

* **5**: ``block_indices_mask`` - tensor of type *T_MASK* with the same shape as
  ``block_indices``. A ``false`` (or zero) entry marks a padding slot that must not
  contribute to the output, used when the number of actually-relevant candidate blocks is
  ragged across query blocks. **Optional.**

* **6**: ``scale`` - a scalar or single-element 1D tensor of type *T*, an alternative scale
  factor instead of the default ``1 / sqrt(E)``. **Optional.**

**Outputs**

* **1**: the result of block-sparse scaled dot-product attention, a tensor of type *T* and
  shape ``[B, H, L, Ev]``.

**Types**

* *T*: any supported floating-point type.

* *T_IDX*: any supported integer type.

* *T_MASK*: ``boolean``. ``u8`` is also accepted as an equivalent alternative: some plugins
  normalize boolean tensors to ``u8`` storage ahead of the operations that consume them (for
  example via a generic boolean-to-``u8`` precision conversion pass), so a ``u8`` tensor
  containing only ``0``/``1`` values is accepted with the exact same semantics as a
  ``boolean`` one.

**Dimensions**

* ``B`` - batch size.

* ``H`` - number of query (and output) attention heads.

* ``Hk`` - number of key/value attention heads: either ``H`` or ``1`` (broadcast).

* ``Hb`` - number of ``block_indices``/``block_indices_mask`` heads: either ``H`` or ``1``
  (broadcast).

* ``L`` - query sequence length. Must be evenly divisible by *block_size*.

* ``S`` - key/value sequence length. Must be evenly divisible by *block_size*.

* ``E`` - embedding (head) size of query and key.

* ``Ev`` - embedding (head) size of value and of the output.

* ``k_blocks`` - number of key/value blocks selected per query block.

Unlike *ScaledDotProductAttention*, *BlockSparseAttention* requires exactly one batch
dimension and exactly one head dimension (rank-4 ``query``/``key``/``value``); it does not
support the arbitrary number of leading batch dimensions that *ScaledDotProductAttention*
allows.

**Examples**

*Example 1: Non-causal block-sparse attention with an explicit padding mask*

.. code-block:: xml
   :force:

    <layer id="42" name="BlockSparseAttention_0" type="BlockSparseAttention" version="opset17">
        <data block_size="2" causal="false" />
        <input>
            <!-- query: B=1, H=2, L=4 (2 query blocks), E=8 -->
            <port id="0" precision="FP32">
                <dim>1</dim>
                <dim>2</dim>
                <dim>4</dim>
                <dim>8</dim>
            </port>
            <!-- key: B=1, Hk=2, S=6 (3 key/value blocks), E=8 -->
            <port id="1" precision="FP32">
                <dim>1</dim>
                <dim>2</dim>
                <dim>6</dim>
                <dim>8</dim>
            </port>
            <!-- value: B=1, Hk=2, S=6, Ev=8 -->
            <port id="2" precision="FP32">
                <dim>1</dim>
                <dim>2</dim>
                <dim>6</dim>
                <dim>8</dim>
            </port>
            <!-- block_indices: B=1, Hb=2, num_q_blocks=2, k_blocks=2 -->
            <port id="3" precision="I32">
                <dim>1</dim>
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
            <!-- block_indices_mask: same shape as block_indices -->
            <port id="4" precision="BOOL">
                <dim>1</dim>
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </input>
        <output>
            <port id="5" precision="FP32">
                <dim>1</dim>
                <dim>2</dim>
                <dim>4</dim>
                <dim>8</dim>
            </port>
        </output>
    </layer>

*Example 2: Causal block-sparse attention, no mask, default scale, broadcast key/value head*

.. code-block:: xml
   :force:

    <layer id="43" name="BlockSparseAttention_1" type="BlockSparseAttention" version="opset17">
        <data block_size="4" causal="true" />
        <input>
            <!-- query: B=2, H=4, L=8 (2 query blocks), E=64 -->
            <port id="0" precision="FP16">
                <dim>2</dim>
                <dim>4</dim>
                <dim>8</dim>
                <dim>64</dim>
            </port>
            <!-- key: B=2, Hk=1 (broadcast to all 4 query heads), S=16 (4 key/value blocks), E=64 -->
            <port id="1" precision="FP16">
                <dim>2</dim>
                <dim>1</dim>
                <dim>16</dim>
                <dim>64</dim>
            </port>
            <!-- value: B=2, Hk=1, S=16, Ev=64 -->
            <port id="2" precision="FP16">
                <dim>2</dim>
                <dim>1</dim>
                <dim>16</dim>
                <dim>64</dim>
            </port>
            <!-- block_indices: B=2, Hb=1, num_q_blocks=2, k_blocks=2 -->
            <port id="3" precision="I64">
                <dim>2</dim>
                <dim>1</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </input>
        <output>
            <port id="4" precision="FP16">
                <dim>2</dim>
                <dim>4</dim>
                <dim>8</dim>
                <dim>64</dim>
            </port>
        </output>
    </layer>
