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
- **By-token quantized caches** (u8, u4): Direct copy of quantized data with scale/zero-point parameters.
- **By-channel quantized caches** (u8, u4): Requires dequantization, token movement in float space, and requantization
  with updated per-channel scale/zero-point parameters when the block's data distribution changes.

Signed ``int8`` KV cache is not supported by this operation.

**Paged memory organization**

The KV cache is organized into non-contiguous pages (blocks). Each physical block stores a fixed number of token states
(``block_size``). The memory layout depends on the quantization mode within different plugins.

**Reordering algorithm**

For each update operation ``(src_token_idx, dst_token_idx)``:

1. Convert logical token indices to ``(block_idx, token_in_block)`` pairs.
2. copy operations:

   - **By-channel quantized**: Dequantize entire block → move tokens in float space → requantize with new statistics.
   - **By-token quantized or non-quantized**: Direct data copy within the block. Parameters, if any, should be copied based on how they are stored (interleaved or separate).

.. code-block:: py
    :force:

    # Parallel processing across sequences
    for s in parallel(num_sequences):
        # Get update operations for this sequence
        op_start = block_update_indices_begins[s]
        op_end   = block_update_indices_begins[s + 1]

        # Get block mapping for this sequence
        blk_start = block_indices_begins[s]
        blk_end   = block_indices_begins[s + 1]
        seq_blocks = block_indices[blk_start:blk_end]

        # Process operations in batches with same (src_block, dst_block) pair
        op_idx = op_start
        while op_idx < op_end:
            src_logical = block_update_indices[op_idx * 2 + 0]
            dst_logical = block_update_indices[op_idx * 2 + 1]

            # Convert to (block, token) pairs
            src_block_local = src_logical // block_size
            dst_block_local = dst_logical // block_size
            src_block = seq_blocks[src_block_local]
            dst_block = seq_blocks[dst_block_local]

            # Collect all operations with same (src_block, dst_block) into batch
            batch_end = op_idx + 1
            while batch_end < op_end:
                next_src = block_update_indices[batch_end * 2 + 0]
                next_dst = block_update_indices[batch_end * 2 + 1]
                next_src_block = seq_blocks[next_src // block_size]
                next_dst_block = seq_blocks[next_dst // block_size]
                if next_src_block != src_block or next_dst_block != dst_block:
                    break
                batch_end += 1

            # Process batch: [op_idx, batch_end)
            same_block = (src_block == dst_block)

            # Parallel processing across heads
            for h in parallel(num_kv_heads):
                if by_channel_quantized:
                    if same_block:
                        # Same block: direct copy without requantization
                        for op in range(op_idx, batch_end):
                            src_token = block_update_indices[op * 2 + 0] % block_size
                            dst_token = block_update_indices[op * 2 + 1] % block_size
                            key_cache[src_block, h, dst_token] = key_cache[src_block, h, src_token]
                            value_cache[src_block, h, dst_token] = value_cache[src_block, h, src_token]
                    else:
                        # Cross-block: dequantize dst once, copy all, requantize once
                        # Key cache
                        dst_block_data = dequantize_by_channel(key_cache[dst_block, h])
                        for op in range(op_idx, batch_end):
                            src_token = block_update_indices[op * 2 + 0] % block_size
                            dst_token = block_update_indices[op * 2 + 1] % block_size
                            src_token_data = dequantize_by_channel(key_cache[src_block, h, src_token])
                            dst_block_data[dst_token] = src_token_data
                        key_cache[dst_block, h] = quantize_by_channel(dst_block_data)

                        # Value cache (same process)
                        dst_block_data = dequantize_by_channel(value_cache[dst_block, h])
                        for op in range(op_idx, batch_end):
                            src_token = block_update_indices[op * 2 + 0] % block_size
                            dst_token = block_update_indices[op * 2 + 1] % block_size
                            src_token_data = dequantize_by_channel(value_cache[src_block, h, src_token])
                            dst_block_data[dst_token] = src_token_data
                        value_cache[dst_block, h] = quantize_by_channel(dst_block_data)
                else:
                    # Non-by-channel: direct copy for all operations in batch
                    for op in range(op_idx, batch_end):
                        src_token = block_update_indices[op * 2 + 0] % block_size
                        dst_token = block_update_indices[op * 2 + 1] % block_size
                        key_cache[dst_block, h, dst_token] = key_cache[src_block, h, src_token]
                        value_cache[dst_block, h, dst_token] = value_cache[src_block, h, src_token]

            # Move to next batch
            op_idx = batch_end


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
  Signed ``int8`` KV cache is not supported.

* **T_IND**: ``int32``.
  Index type for all index tensors.
