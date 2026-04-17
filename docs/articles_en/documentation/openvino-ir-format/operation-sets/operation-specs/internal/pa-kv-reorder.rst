.. {#openvino_docs_ops_internal_PaKVReorder}

PaKVReorder
===========

.. meta::
  :description: Learn about PaKVReorder - a key-value cache reordering operation for paged attention in autoregressive models.

**Versioned name**: *PaKVReorder*

**Category**: *Internal*

**Short description**:
The *PaKVReorder* operation reorders tokens within the key-value (KV) cache blocks used by paged attention mechanisms,
enabling efficient token management for speculative decoding after draft token verification
in autoregressive language models.

**Detailed description**

*PaKVReorder* processes paged KV cache blocks by moving tokens from source positions to destination positions
according to the provided update indices. The operation is kv manipulation for speculative decoding, when draft tokens are verified, accepted tokens need to be kept while rejected ones
are discarded, requiring reordering of the KV cache to maintain sequence consistency.

The KV cache may use quantization (by-channel or by-token) for memory efficiency. The operation handles:

- **Non-quantized caches** (fp32, bf16, fp16): Direct memory copy of token embeddings.
- **By-token quantized caches** (i8, u8, u4): Direct copy of quantized data with scale/zero-point parameters.
- **By-channel quantized caches** (u8, u4): Requires dequantization, token movement in float space, and requantization
  with updated per-channel scale/zero-point parameters when the block's data distribution changes.

**Paged memory organization**

The KV cache is organized into non-contiguous pages (blocks). Each physical block stores a fixed number of token states
(``block_size``). The memory layout depends on the quantization mode within different plugins.

**Reordering algorithm**

For each update operation ``(src_token_idx, dst_token_idx)``:

1. Convert logical token indices to ``(block_idx, token_in_block)`` pairs.
2. **copy operations** (``src_block == dst_block``):

   - **By-channel quantized**: Dequantize entire block → move tokens in float space → requantize with new statistics.
   - **By-token quantized**: Direct data copy within the block. Parameters been copied based on how they are stored (interleaved or separate).

.. code-block:: py
    :force:

    for s in range(num_sequences):
    # Get update operations for this sequence
    op_start = block_update_indices_begins[s]
    op_end   = block_update_indices_begins[s + 1]

    # Get block mapping for this sequence
    blk_start = block_indices_begins[s]
    blk_end   = block_indices_begins[s + 1]
    seq_blocks = block_indices[blk_start:blk_end]

    for op_idx in range(op_start, op_end):
        src_logical = block_update_indices[op_idx * 2 + 0]
        dst_logical = block_update_indices[op_idx * 2 + 1]

        # Convert logical indices to (block, token) pairs
        src_block_local = src_logical // block_size
        dst_block_local = dst_logical // block_size
        src_token = src_logical % block_size
        dst_token = dst_logical % block_size

        # Get physical block indices
        src_block = seq_blocks[src_block_local]
        dst_block = seq_blocks[dst_block_local]

        # Process each head independently
        for h in range(num_kv_heads):
            if by_channel_quantized:
                # Dequantize source token
                src_token_data = dequantize_by_channel(key_cache[src_block, h, src_token])
                # Dequantize destination block
                dst_block_data = dequantize_by_channel(key_cache[dst_block, h])
                # Copy token in float space
                dst_block_data[dst_token] = src_token_data
                # Requantize destination block
                key_cache[dst_block, h] = quantize_by_channel(dst_block_data)
            else:
                # Direct copy
                key_cache[dst_block, h, dst_token] = key_cache[src_block, h, src_token]

            # Repeat for value_cache


**Attributes**: *PaKVReorder* operation has no attributes.


**Inputs**

* **0**: ``key_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, num_kv_heads, block_size, key_hidden]`` for non-quantized,
  or ``[num_blocks, num_kv_heads, block_size + params_size, key_hidden]`` for by-channel quantization.
  Paged key cache storing attention keys. Updated in-place during reordering. **Required.**

* **1**: ``value_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, num_kv_heads, block_size, value_hidden]`` for non-quantized,
  or ``[num_blocks, num_kv_heads, block_size + params_size, value_hidden]`` for by-channel quantization.
  Paged value cache storing attention values. Updated in-place during reordering. **Required.**

* **2**: ``block_indices``
  A 1D tensor of type *T_IND* with shape ``[total_num_blocks]``.
  Physical block indices for all sequences. Maps logical block indices to physical block addresses in the cache.
  ``total_num_blocks = block_indices_begins[-1]``. **Required.**

* **3**: ``block_indices_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Splits ``block_indices`` among sequences. Sequence ``s`` uses blocks from
  ``block_indices[block_indices_begins[s] : block_indices_begins[s+1]]``. **Required.**

* **4**: ``block_update_indices``
  A 1D tensor of type *T_IND* with shape ``[total_updates * 2]``.
  Pairs of ``(src_logical_token_idx, dst_logical_token_idx)`` specifying token movements.
  Each pair represents one reorder operation. Indices are logical (sequence-relative) token positions.
  ``total_updates = block_update_indices_begins[-1]``. **Required.**

* **5**: ``block_update_indices_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Splits ``block_update_indices`` among sequences. Sequence ``s`` has operations from
  ``block_update_indices[block_update_indices_begins[s]*2 : block_update_indices_begins[s+1]*2]``.
  **Required.**


**Outputs**

* **0**: ``status``
  A 1D tensor of type *u8* with shape ``[1]``.
  Status indicator (always 0 for success). The actual output is the in-place modification of ``key_cache``
  and ``value_cache``.


**Types**

* **T**: ``float32``, ``bfloat16``, ``float16``, ``uint8``, ``uint4``.
  Determines the data type of KV cache. Quantized types (``uint8``, ``uint4``) require scale/zero-point parameters.

* **T_IND**: ``int32``.
  Index type for all index tensors.


**Example: Speculative decoding token acceptance**

.. code-block:: py
    :force:

    # Scenario: 3 draft tokens proposed, 2 accepted (tokens 47, 48), 1 rejected (token 49)
    # Need to remove token 49 and keep tokens 47, 48

    num_sequences = 1
    block_size = 32

    # Token 47, 48, 49 are in logical block 1 (positions 15, 16, 17 within block)
    # After rejection, token 49's position should be filled by shifting or left empty

    # Option 1: Compact by moving token 48 to position of token 49 (not used here)
    # Option 2: Simply mark positions as valid/invalid (GenAI tracks valid length)

    # For this example, we reorder to move accepted tokens together:
    block_indices = [5, 8]  # Sequence 0 uses physical blocks 5, 8
    block_indices_begins = [0, 2]

    # No reordering needed if we just adjust the sequence length
    # But if we want to compact: move token 48 (logical 48) to fill gaps
    block_update_indices = []  # Empty if no compaction
    block_update_indices_begins = [0, 0]

    # Run PaKVReorder
    output = PaKVReorder(
        key_cache, value_cache,
        block_indices, block_indices_begins,
        block_update_indices, block_update_indices_begins
    )
