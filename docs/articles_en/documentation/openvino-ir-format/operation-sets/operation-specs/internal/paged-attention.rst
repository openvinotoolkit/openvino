.. {#openvino_docs_ops_internal_PagedAttention}

PagedAttention
==============

.. meta::
   :description: Learn about PagedAttention - an attention operator for speculative decoding with in-kernel cache eviction.

**Versioned name**: *PagedAttention*

**Category**: *Sequence processing*

**Short description**: *PagedAttention* implements an optimized attention operator using a block-based KV cache system to enable speculative decoding, efficient memory reuse, and scalable transformer inference.

---

**Detailed description**:

PagedAttention enables high-performance transformer inference by implementing blockwise key-value (KV) memory layouts, dynamic input chunking, and optional cache eviction logic. It efficiently attends over a combination of past and present tokens across multiple sequences with varying lengths and independently managed memory blocks.

---

**Algorithm Explanation**

At the core, attention is computed as:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Bias}\right)V
\]

**With enhancements:**

- **KV Cache Layout**: Keys and values are stored in *blocks* of shape `[block_size, head_size]` indexed by `block_indices`. Each sequence may use different blocks for its context.
- **Subsequence Tokens**: The `subsequence_begins` input defines token start offsets for each sequence in the token batch.
- **Block Insertion**: When new tokens are processed, their `K` and `V` are inserted into KV cache blocks at locations determined by `block_indices`, updating these blocks in-place.
- **Optional Eviction Logic**: Controlled externally or internally (if enabled via kernel logic), older blocks may be replaced to maintain memory limits.

**Note:** `block_indices` does not track which blocks are empty or free. It assumes the external memory manager has already assigned valid, available blocks for insertion.

---

**KV Cache Insertion – Pseudocode**

.. code-block:: python

    def insert_into_cache(key_cache, value_cache, key, value, block_indices, subsequence_begins, block_size):
        """
        Insert new key/value tensors into cache blocks.
        Assumes all block_indices have already been allocated and are valid.
        """
        token_idx = 0
        for seq_id, block_begin in enumerate(subsequence_begins[:-1]):
            seq_token_count = subsequence_begins[seq_id + 1] - block_begin
            # Blocks assigned to this sequence
            assigned_blocks = block_indices[block_begin : block_begin + (seq_token_count + block_size - 1) // block_size]

            for i in range(seq_token_count):
                block_id = assigned_blocks[i // block_size]
                offset = i % block_size
                key_cache[block_id, :, offset, :] = key[token_idx]
                value_cache[block_id, :, offset, :] = value[token_idx]
                token_idx += 1

---

**Inputs**

> Here is a brief description of each input parameter:

1. **query** – Query tensor `[batch_size_in_tokens, num_heads * head_size]`, type *T*.  
   `batch_size_in_tokens` is the total number of tokens across all sequences in the current batch.
2. **key** – Key tensor `[batch_size_in_tokens, num_kv_heads * head_size]`, type *T*
3. **value** – Value tensor `[batch_size_in_tokens, num_kv_heads * head_size]`, type *T*
4. **key_cache** – Key cache `[num_blocks, num_kv_heads, block_size, head_size]`, type *T*.
5. **value_cache** – Same shape and type as key_cache
6. **past_lens** – `[batch_size_in_sequences]`, type `int32`. Number of previously cached tokens per sequence.
7. **subsequence_begins** – `[batch_size_in_sequences + 1]`, type `int32`. Offsets marking start of each sequence's tokens.
8. **block_indices** – `[num_blocks]`, type `int32`. Maps tokens to cache blocks.
9. **block_indices_begins** – `[batch_size_in_sequences + 1]`, type `int32`. Start indices of block mapping per sequence.
10. **scale** – scalar, type *T* (optional). Scaling factor for attention.
11. **sliding_window** – scalar, type `int32` (optional). Limits context size per sequence.
12. **alibi_slopes** – `[num_kv_heads]`, type *T* (optional). ALiBi bias slopes per head.
13. **rotated_block_indices** – `[num_rotated_blocks]`, type `int32` (optional). Used for rotary position embedding.
14. **rotation_deltas** – `[num_rotated_blocks, BLOCK_SIZE || 1]`, type `int32` (optional). Position shift values.
15. **rotation_trig_lut** – `[M, head_size]`, type *T* (optional). Lookup table for sinusoidal rotary embedding.
16. **score_aggregation_window** – scalar, type `int32` (optional). Window size for attention score aggregation.

---

**Outputs**

1. **Output** – `[batch_size_in_tokens, num_heads * head_size]`, type *T*
2. **scores** – `[sum(past_lens) + batch_size_in_tokens]`, type *T* (optional)

---

**Types**

* *T*: any floating point type (e.g., `f16`, `f32`, `bf16`)

---

**Example**

.. code-block:: xml
   :force:

    <layer id="42" name="paged_attention" type="PagedAttention" version="ie_internal_opset">
        <input>
            <port id="0" precision="FP32"><dim>-1</dim><dim>128</dim></port>
            <port id="1" precision="FP32"><dim>-1</dim><dim>128</dim></port>
            <port id="2" precision="FP32"><dim>-1</dim><dim>128</dim></port>
            <port id="3" precision="FP32"><dim>-1</dim><dim>8</dim><dim>16</dim><dim>16</dim></port>
            <port id="4" precision="FP32"><dim>-1</dim><dim>8</dim><dim>16</dim><dim>16</dim></port>
            <port id="5" precision="I32"><dim>-1</dim></port>
            <port id="6" precision="I32"><dim>-1</dim></port>
            <port id="7" precision="I32"><dim>-1</dim></port>
            <port id="8" precision="I32"><dim>-1</dim></port>
            <port id="9" precision="FP32"/>
            <port id="10" precision="I32"/>
            <port id="11" precision="FP32"><dim>8</dim></port>
            <port id="12" precision="I32"><dim>-1</dim></port>
            <port id="13" precision="I32"><dim>-1</dim><dim>1</dim></port>
            <port id="14" precision="FP32"><dim>-1</dim><dim>128</dim></port>
            <port id="15" precision="I32"/>
        </input>
        <output>
            <port id="16" precision="FP32"><dim>-1</dim><dim>128</dim></port>
            <port id="17" precision="FP32"><dim>-1</dim></port>
        </output>
    </layer>
