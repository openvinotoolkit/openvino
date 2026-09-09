.. {#openvino_docs_ops_internal_PagedSelectiveSSM}

PagedSelectiveSSM
=================

.. meta::
  :description: Learn about PagedSelectiveSSM - a paged SelectiveSSM (Mamba2 selective state-space) operation for memory-efficient LLM inference with grouped projections.

**Versioned name**: *PagedSelectiveSSM*

**Category**: *Internal*

**Short description**: *PagedSelectiveSSM* implements the paged variant of the
SelectiveSSM (Mamba2 selective state-space) recurrence with grouped projections for
memory-efficient batched LLM inference.

**Detailed description**:

The *PagedSelectiveSSM* operation is the paged variant of the SelectiveSSM
(`arXiv:2405.21060 <https://arxiv.org/abs/2405.21060>`__) selective state-space
recurrence. It processes tokens from multiple sequences packed into a single batch and
manages the recurrent SSM state (a per-head ``[head_dim, state_size]`` matrix) using a
paged block table, enabling non-contiguous memory allocation across sequences.

The operation takes the raw (un-discretized) parameters ``A`` (log-decay rates), ``dt``
(time steps), ``B`` (input projection), ``x`` (input hidden states) and ``C`` (output
projection). The time discretization of ``A`` and ``B`` is computed per token before the
recurrent update (``dA = exp(A * dt)`` and ``dtB = dt * B``). The ``B`` and ``C``
projections are grouped and shared across heads: each head ``h`` reads the group
``g = h // heads_per_group`` where ``heads_per_group = num_heads // num_groups``.

For each token ``t`` in each sequence, the operation applies the following recurrent
update for every head ``h`` (``0 .. num_heads - 1``), with the corresponding group index
``g = h // heads_per_group``:

.. code-block:: py
   :force:

    # Grouped projection mapping: each head h reads the group g
    heads_per_group = num_heads // num_groups
    g = h // heads_per_group

    # Per-token time discretization
    dA = torch.exp(A[h] * dt[t, h])          # scalar
    dtB = dt[t, h] * B[t, g]                  # shape: [state_size]

    # dBx = x_t outer dtB
    dBx = outer(x[t, h], dtB)                 # shape: [head_dim, state_size]

    # state_t = state_{t-1} * dA + dBx
    # S[h]: recurrent state, shape [head_dim, state_size]
    S[h] = S[h] * dA + dBx

    # y_t = reduce_sum(state_t * C_t, axis=state_size)
    output[t, h] = S[h] @ C[t, g]             # shape: [head_dim]

For new input sequence the initial recurrent state should be zeroed. User can set own state if needed.
It is updated token by token within each sequence in causal order.
The operation caches intermediate states at regular intervals
(controlled per sequence by ``cache_interval``) into the paged ``recurrent_state_table``, allowing
efficient prefill replay and incremental decode.

**Paged memory management**

The recurrent state table is organized as non-contiguous pages (blocks). Each block stores one
complete state snapshot for all heads at a particular token position in the sequence.

For sequence ``s``, the assigned physical block indices are
``la_block_indices[la_block_indices_begins[s] : la_block_indices_begins[s+1]]``.
These indices address rows in ``recurrent_state_table``. The first block stores the state after
``cache_interval[s]`` tokens, the second after ``2 * cache_interval[s]`` tokens, and so on.
When ``cache_interval[s] <= 0``, no state caching is performed for that sequence.

The ``num_processed_tokens[s]`` value indicates how many tokens have already been processed for sequence
``s``. Denote ``num_current_tokens[s]`` as the number of current tokens to process.
It can be computed as: ``subsequence_begins[s+1] - subsequence_begins[s]``.
Then `N`, the number of blocks for writing, is computed as:
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


**Attributes**

*PagedSelectiveSSM* operation has no attributes.

**Inputs**

* **1**: ``A`` - Tensor of type *T* and shape ``[num_heads]``.
  Per-head (negative) log-decay rates used to compute the discretized state transition
  ``dA``. **Required.**

* **2**: ``dt`` - Tensor of type *T* and shape ``[batch_size_in_tokens, num_heads]``.
  Per-token, per-head time steps used for discretization. **Required.**

* **3**: ``B`` - Tensor of type *T* and shape ``[batch_size_in_tokens, num_groups, state_size]``.
  Grouped input projection for all tokens in the batch. Shared across heads within a group.
  **Required.**

* **4**: ``x`` - Tensor of type *T* and shape ``[batch_size_in_tokens, num_heads, head_dim]``.
  Input hidden states for all tokens in the batch. **Required.**

* **5**: ``C`` - Tensor of type *T* and shape ``[batch_size_in_tokens, num_groups, state_size]``.
  Grouped output projection for all tokens in the batch. Shared across heads within a group.
  **Required.**

* **6**: ``recurrent_state_table`` - Tensor of type *T_cache* and shape
  ``[num_blocks, num_heads, head_dim, state_size]``.
  Paged table of recurrent state snapshots. Each row is one block storing a complete state for
  all heads at a cached token position. This tensor is updated in place during execution.
  The initial state before any tokens are processed is an all-zeros tensor. **Required.**

* **7**: ``subsequence_begins`` - Tensor of type *T_IND* and shape ``[batch_size_in_sequences + 1]``.
  Start indices of each sequence's tokens in the flattened token batch (0-th dimension of
  ``dt``, ``B``, ``x``, ``C``). The tokens of sequence ``s`` span
  ``[subsequence_begins[s], subsequence_begins[s+1])``. **Required.**

* **8**: ``la_block_indices`` - Tensor of type *T_IND* and shape ``[num_blocks]``.
  Physical block row indices into ``recurrent_state_table``, concatenated across all sequences.
  For example, ``[0, 1, 3, 2, 4]`` with five blocks. **Required.**

* **9**: ``la_block_indices_begins`` - Tensor of type *T_IND* and shape
  ``[batch_size_in_sequences + 1]``.
  Splits ``la_block_indices`` among sequences. The block indices for sequence ``s`` are
  ``la_block_indices[la_block_indices_begins[s] : la_block_indices_begins[s+1]]``.
  For example, ``la_block_indices = [0, 1, 3, 2, 4]`` and ``la_block_indices_begins = [0, 3, 5]``
  means sequence 0 uses blocks ``[0, 1, 3]`` and sequence 1 uses blocks ``[2, 4]``. **Required.**

* **10**: ``num_processed_tokens`` - Tensor of type *T_IND* and shape ``[batch_size_in_sequences]``.
  Number of tokens already processed for each sequence. Used together with the cached states
  in ``recurrent_state_table`` to determine the starting recurrent state for each sequence.
  **Required.**

* **11**: ``cache_interval`` - Tensor of type *T_IND* and shape ``[batch_size_in_sequences]``.
  Interval (in tokens) at which the recurrent state is saved into a block of
  ``recurrent_state_table`` for each sequence. A value ``<= 0`` disables caching for that
  sequence. **Required.**

**Outputs**

* **1**: ``output`` - Tensor of type *T* and shape
  ``[batch_size_in_tokens, num_heads, head_dim]``.
  Per-token, per-head output vectors produced by contracting the updated recurrent state
  with ``C``.

.. note::

   ``recurrent_state_table`` (input 6) is updated in place as a side effect. The initial
   recurrent state is an all-zeros tensor, consistent with the StatefulCausalConv1D convention.

.. note::

   This operation uses grouped projections. The number of heads sharing each group is
   ``heads_per_group = num_heads // num_groups``. Each group of ``B``/``C`` is shared by
   ``heads_per_group`` consecutive heads, with the mapping ``g = h // heads_per_group``.

**Types**

* *T*: any floating-point type.

* *T_IND*: ``int32``.

* *T_cache* - cache precision; may differ from *T*.
  Allowed for ``recurrent_state_table`` (input 6): ``f16``, ``f32``, ``bf16``.


**Example**

.. code-block:: xml
   :force:

   <layer ... type="PagedSelectiveSSM" ...>
       <input>
           <port id="0"> <!-- `A` -->
               <dim>8</dim>
           </port>
           <port id="1"> <!-- `dt` -->
               <dim>6</dim>
               <dim>8</dim>
           </port>
           <port id="2"> <!-- `B` -->
               <dim>6</dim>
               <dim>2</dim>
               <dim>128</dim>
           </port>
           <port id="3"> <!-- `x` -->
               <dim>6</dim>
               <dim>8</dim>
               <dim>64</dim>
           </port>
           <port id="4"> <!-- `C` -->
               <dim>6</dim>
               <dim>2</dim>
               <dim>128</dim>
           </port>
           <port id="5"> <!-- `recurrent_state_table` -->
               <dim>5</dim>
               <dim>8</dim>
               <dim>64</dim>
               <dim>128</dim>
           </port>
           <port id="6"> <!-- `subsequence_begins` -->
               <dim>3</dim>
           </port>
           <port id="7"> <!-- `la_block_indices` -->
               <dim>5</dim>
           </port>
           <port id="8"> <!-- `la_block_indices_begins` -->
               <dim>3</dim>
           </port>
           <port id="9"> <!-- `num_processed_tokens` -->
               <dim>2</dim>
           </port>
           <port id="10"> <!-- `cache_interval` -->
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="11"> <!-- `output` -->
               <dim>6</dim>
               <dim>8</dim>
               <dim>64</dim>
           </port>
       </output>
   </layer>
