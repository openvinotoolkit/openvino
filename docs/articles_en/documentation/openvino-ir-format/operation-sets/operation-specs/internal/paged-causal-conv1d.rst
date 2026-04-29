.. {#openvino_docs_ops_internal_PagedCausalConv1D}

PagedCausalConv1D
=================

.. meta::
  :description: Learn about PagedCausalConv1D - a paged causal 1D convolution operation for time-series and autoregressive models.

**Versioned name**: *PagedCausalConv1D*

**Category**: *Internal*

**Short description**:
The *PagedCausalConv1D* operation performs a causal 1D grouped convolution over a batch of token sequences,
maintaining the convolution state in paged memory blocks. The causal 1D-convolution is a 1D convolution layer where each output
at time step ``t`` depends only on inputs from time ``<=t`` (not future values),
making it suitable for time-series and autoregressive models. 

**Detailed description**

*PagedCausalConv1D* processes a flat batch of tokens that may belong to multiple independent sequences.
The token sequences are described by ``subsequence_begins``. Paged memory uses a fixed ``BLOCK_SIZE=1``,
meaning each block in ``conv_state_table`` stores exactly one convolution state snapshot of shape ``[hidden_size, kernel_size]``.
For each sequence, the operation:

1. Loads the current convolution state (a window of the last `kernel_size` input vectors) from paged memory using the block table.

2. For each token, shifts the state window, inserts the new token, and applies a grouped causal 1D convolution to produce the output embedding.

3. Caches intermediate states to paged memory blocks at intervals controlled by `cache_interval`  
   (used during prefill to support prefix caching and chunked prefill).

4. Saves the final state for each sequence into the last assigned block.


For new input sequence the initial state should be zeroed. User can set own state if needed.
Paged memory management allows states to be shared across sequences (prefix caching) and allocated on demand.


**Paged memory management**

The convolutional state table is organized as non-contiguous pages (blocks). Each block stores one
complete state snapshot at a particular token position in the sequence.

For sequence ``s``, the assigned physical block indices are
``la_block_indices[la_block_indices_begins[s] : la_block_indices_begins[s+1]]``.
These indices address rows in ``conv_state_table``. The first block stores the state after
``cache_interval[s]`` tokens, the second after ``2 * cache_interval[s]`` tokens, and so on.
When ``cache_interval[s] <= 0``, no state caching is performed for that sequence.

The ``num_processed_tokens[s]`` value indicates how many tokens have already been processed for sequence
``s``. Denote ``num_current_tokens[s]`` as the number of current tokens to process.
It can be computed as: ``subsequence_begins[s+1] - subsequence_begins[s]``.
Then `N`, the number of blocks for writing required for sequence ``s``, is computed as:
``N = ceil((num_processed_tokens[s] % cache_interval[s] + num_current_tokens[s]) / cache_interval[s])``
Let the blocks passed through `la_block_indices` be indexed as block `0, 1, ..., N`.
Cases for reading and updating blocks:

1. **Prefill with no past**  
   Read from block 0 and write to blocks 1...N.  
   Block 0 and block 1 refer to the same block, so block 0 is updated in-place.

2. **Prefill with `num_processed_tokens[s] % cache_interval[s] == 0`**  
   Read from block 0 and write to blocks 1...N.  
   Block 0 and block 1 refer to different blocks.

3. **Prefill with `num_processed_tokens[s] % cache_interval[s] != 0`**  
   Read from block 0 and write to blocks 1...N.  
   Block 0 and block 1 refer to the same block, so block 0 is updated in-place.

4. **Decode with `num_processed_tokens[s] % cache_interval[s] == 0`**  
   Read from block 0 and write to block 1.  
   Block 0 and block 1 refer to different blocks.

5. **Decode with `num_processed_tokens[s] % cache_interval[s] != 0`**  
   Read from block 0 and write to block 1.  
   Block 0 and block 1 refer to the same block, so block 0 is updated in-place.


.. code-block:: py
    :force:

    for s in range(batch_size_in_sequences):
        token_start = subsequence_begins[s]
        token_end   = subsequence_begins[s + 1]
        seq_len     = token_end - token_start

        # Physical block indices for this sequence
        blk_start  = la_block_indices_begins[s]
        blk_end    = la_block_indices_begins[s + 1]
        seq_blocks = la_block_indices[blk_start:blk_end]  # list of physical block indices

        # Load current convolution state from the last assigned block.
        # state shape: [hidden_size, kernel_size], where state[:, -1] is
        # the most recently seen input and state[:, 0] is the oldest.
        # Initially zero if the sequence has not been processed before.
        state = conv_state_table[seq_blocks[-1]]  # [hidden_size, kernel_size]

        for t in range(seq_len):
            x = input_embeds[token_start + t]  # [hidden_size]

            # Shift state left and append the new token on the right
            state = concat([state[:, 1:], x[:, newaxis]], axis=-1)  # [hidden_size, kernel_size]

            # Grouped causal 1D convolution output
            output_embeds[token_start + t] = grouped_conv1d(state, conv_weight, conv_bias)

            # Cache state at regular intervals (for prefix caching / chunked prefill)
            abs_pos = num_processed_tokens[s] + t  # 0-based absolute position of the token
            if cache_interval[s] > 0 and (abs_pos + 1) % cache_interval[s] == 0:
                blk = (abs_pos + 1) // cache_interval[s] - 1
                conv_state_table[seq_blocks[blk]] = copy(state)

        # Persist the final state for this sequence into the last block
        conv_state_table[seq_blocks[-1]] = state

Where ``grouped_conv1d`` computes a standard grouped (or depthwise when ``group_size == hidden_size``) convolution
over the state window. Here ``group_size`` is the number of input channels per group, ``groups = hidden_size // group_size``
is the total number of groups (equivalent to ``out_channels // (hidden_size // group_size)`` given
the constraint ``out_channels == hidden_size``), and ``out_channels`` must equal ``hidden_size``
(as required by the output shape). The weight tensor second dimension is ``hidden_size // group_size`` (channels per group):

.. code-block:: py
    :force:

    def grouped_conv1d(state, conv_weight, conv_bias):
        # state:       [hidden_size, kernel_size]
        # conv_weight: [out_channels, hidden_size // group_size, kernel_size]
        #              where out_channels == hidden_size (constraint for this operation)
        # conv_bias:   [out_channels]
        # returns:     [out_channels]
        #
        # group_size:  number of input channels per convolution group
        #              (equals hidden_size for depthwise; 1 for channel-wise)
        groups       = hidden_size // group_size   # total number of groups (== out_channels / ic_per_group, given out_channels == hidden_size)
        ic_per_group = hidden_size // groups       # == group_size
        output = zeros(out_channels)
        for oc in range(out_channels):
            g          = oc * groups // out_channels  # group index for output channel oc
            ic_start   = g * ic_per_group
            ic_end     = ic_start + ic_per_group
            output[oc] = sum(conv_weight[oc, :, :] * state[ic_start:ic_end, :]) + conv_bias[oc]
        return output


**Attributes**: *PagedCausalConv1D* operation has no attributes.


**Inputs**

* **0**: ``input_embeds``
  A 2D tensor of type *T* with shape ``[batch_size_in_tokens, hidden_size]``.
  Input token embeddings from all sequences in the batch. **Required.**

* **1**: ``conv_state_table``
  A 3D tensor of type *T* with shape ``[num_blocks, hidden_size, kernel_size]``.
  Paged block table holding the convolution cache states. The paged memory block size is fixed at ``BLOCK_SIZE=1``,
  meaning each physical block stores exactly one convolution state of shape ``[hidden_size, kernel_size]``,
  representing the last ``kernel_size`` input vectors seen by the corresponding sequence.
  ``num_blocks`` equals the total number of blocks allocated across all sequences (i.e. ``la_block_indices_begins[-1]``).
  The table is updated in-place: during prefill by the plugin, during decoding by GenAI. Initially all states are zero tensors.
  **Required.**

* **2**: ``conv_weight``
  A 3D tensor of type *T* with shape ``[out_channels, hidden_size / group_size, kernel_size]``.
  Convolution filter weights, where ``group_size`` is the number of input channels per convolution group
  (``1 <= group_size <= hidden_size``). For a depthwise convolution
  ``group_size == hidden_size`` and ``out_channels == hidden_size``, so the shape becomes ``[hidden_size, 1, kernel_size]``.
  The constraint ``out_channels == hidden_size`` is required so that the output shape matches the input shape. **Required.**

* **3**: ``conv_bias``
  A 1D tensor of type *T* with shape ``[out_channels]`` or empty tensor of shape ``[0]``.
  The empty tensor means that bias is not applied.
  Per-output-channel bias added after the convolution. **Required.**

* **4**: ``subsequence_begins``
  A 1D tensor of type *T_IND* with shape ``[batch_size_in_sequences + 1]``.
  Start token indices of each sequence within the flat ``input_embeds`` batch.
  The tokens for sequence ``s`` are ``input_embeds[subsequence_begins[s] : subsequence_begins[s+1]]``.
  The first element is always ``0`` and the last element equals ``batch_size_in_tokens``. **Required.**

* **5**: ``la_block_indices``
  A 1D tensor of type *T_IND* with shape ``[num_blocks]``.
  Physical block indices into ``conv_state_table`` assigned across all sequences,
  where ``num_blocks = la_block_indices_begins[-1]`` is the total number of blocks allocated.
  The logical-to-physical mapping for sequence ``s`` is given
  by ``la_block_indices[la_block_indices_begins[s] : la_block_indices_begins[s+1]]``.
  For example, ``la_block_indices = [0, 1, 3, 2, 4]`` with ``la_block_indices_begins = [0, 3, 5]`` means
  that sequence 0 uses physical blocks ``{0, 1, 3}`` and sequence 1 uses physical blocks ``{2, 4}``.
  The number of blocks is determined by GenAI based on scheduled tokens. **Required.**

* **6**: ``la_block_indices_begins``
  A 1D tensor of type *T_IND* with shape ``[batch_size_in_sequences + 1]``.
  Splits ``la_block_indices`` among sequences.
  The block indices for sequence ``s`` are ``la_block_indices[la_block_indices_begins[s] : la_block_indices_begins[s+1]]``.
  The last block in each sequence's range always holds the sequence's most recent (current) state. **Required.**
  
* **7**: ``num_processed_tokens``
  A 1D tensor of type *T_IND* with shape ``[batch_size_in_sequences]``.
  Number of tokens already processed for each sequence prior to this invocation.
  Used to compute the absolute token position needed for ``cache_interval`` alignment. **Required.**

* **8**: ``cache_interval``
  A 1D tensor of type *T_IND* with shape ``[batch_size_in_sequences]``.
  Per-sequence interval (in tokens) at which the convolution state is snapshotted into a paged block during prefill.
  A value ``<= 0`` disables intermediate state caching for that sequence; only the final state is saved. **Required.**


**Outputs**

* **0**: ``output_embeds``
  A 2D tensor of type *T* with shape ``[batch_size_in_tokens, hidden_size]``.
  Output token embeddings after applying the causal grouped 1D convolution. Has the same layout as ``input_embeds``.
  The constraint ``out_channels == hidden_size`` ensures the output channel count matches the input embedding width.


**Types**

* *T*: any floating point type.
* *T_IND*: ``int32``.


**Example**

The example below illustrates a batch with two sequences (3 and 2 tokens respectively) using
a depthwise causal convolution (``hidden_size == out_channels == 16``, ``kernel_size == 4``).
Five state blocks are allocated in the paged table. Sequence 0 is assigned blocks ``{0, 1, 3}``
and sequence 1 is assigned blocks ``{2, 4}``. The ``cache_interval`` of ``1`` causes each processed
token's state to be cached.

.. code-block:: xml
   :force:

   <layer ... type="PagedCausalConv1D">
       <input>
           <port id="0">   <!-- input_embeds: [batch_size_in_tokens, hidden_size] -->
               <dim>5</dim>
               <dim>16</dim>
           </port>
           <port id="1">   <!-- conv_state_table: [num_blocks, hidden_size, kernel_size] -->
               <dim>5</dim>
               <dim>16</dim>
               <dim>4</dim>
           </port>
           <port id="2">   <!-- conv_weight (depthwise): [out_channels, 1, kernel_size] -->
               <dim>16</dim>
               <dim>1</dim>
               <dim>4</dim>
           </port>
           <port id="3">   <!-- conv_bias: [out_channels] -->
               <dim>16</dim>
           </port>
           <port id="4">   <!-- subsequence_begins: [batch_size_in_sequences+1] -->
               <dim>3</dim>
           </port>
           <port id="5">   <!-- la_block_indices: [num_blocks] -->
               <dim>5</dim>
           </port>
           <port id="6">   <!-- la_block_indices_begins: [batch_size_in_sequences+1] -->
               <dim>3</dim>
           </port>
           <port id="7">   <!-- num_processed_tokens: [batch_size_in_sequences] -->
               <dim>2</dim>
           </port>
           <port id="8">   <!-- cache_interval: [batch_size_in_sequences] -->
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="9">   <!-- output_embeds: [batch_size_in_tokens, hidden_size] -->
               <dim>5</dim>
               <dim>16</dim>
           </port>
       </output>
   </layer>
