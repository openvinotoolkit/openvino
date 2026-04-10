PagedGatedDeltaNet
==================

.. meta::
  :description: Learn about PagedGatedDeltaNet - a paged GatedDeltaNet linear attention operation for memory-efficient LLM inference with grouped-query support.

**Versioned name**: *PagedGatedDeltaNet*

**Category**: *Internal*

**Short description**: *PagedGatedDeltaNet* implements paged GatedDeltaNet linear
attention with grouped-query support for memory-efficient batched LLM inference.

**Detailed description**:

The *PagedGatedDeltaNet* operation is the paged variant of the GatedDeltaNet linear
recurrent attention. It processes tokens from multiple sequences packed into a single batch and
manages the recurrent state (a per-head key-value matrix) using a paged block table, enabling
non-contiguous memory allocation across sequences.

For each token in each sequence, the operation applies the following recurrent update for every
value head ``h_v`` (``0 .. v_num_heads - 1``), where the corresponding query/key head index is
``h_q = h_v // num_groups`` and ``num_groups = v_num_heads // num_heads``:

.. code-block:: py
   :force:

    def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
        """This function is intended to align with the l2norm implementation in the FLA library."""
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    # Optional L2 normalization of query and key (when use_qk_l2norm is True)
    if use_qk_l2norm:
        q[t, h_q] = l2norm(q[t, h_q], dim=-1, q_l2_norm_eps)
        k[t, h_q] = l2norm(k[t, h_q], dim=-1, k_l2_norm_eps)

    # Grouped-query mapping: each value head h_v shares the query/key head h_q
    num_groups = v_num_heads // num_heads
    h_q = h_v // num_groups

    # Retrieve current estimate from state
    # S[h_v]: recurrent state, shape [value_head_dim, key_head_dim]
    r = k[t, h_q] @ S[h_v].tranpose(1, 0)                            # shape: [value_head_dim]

    # Gated delta update
    S[h_v] = gate[t, h_v] * S[h_v] \
           + beta[t, h_v] * outer(k[t, h_q], v[t, h_v] - r).transpose(1, 0)

    # Project with query to get output
    output[t, h_v] = q[t, h_q] @ S[h_v].transpose(1, 0)              # shape: [value_head_dim]

The recurrent state is initialized to an all-zeros tensor. It is updated token by token within
each sequence in causal order. The operation caches intermediate states at regular intervals
(controlled per sequence by ``cache_interval``) into the paged ``recurrent_state_table``, allowing
efficient prefill replay and incremental decode.

**Paged memory management**

The recurrent state table is organized as non-contiguous pages (blocks). Each block stores one
complete state snapshot for all value heads at a particular token position in the sequence.

For sequence ``s``, the assigned physical block indices are
``block_indices[block_indices_begins[s] : block_indices_begins[s+1]]``.
These indices address rows in ``recurrent_state_table``. The first block stores the state after
``cache_interval[s]`` tokens, the second after ``2 * cache_interval[s]`` tokens, and so on.
When ``cache_interval[s] <= 0``, no state caching is performed for that sequence.

The ``past_lens[s]`` value indicates how many tokens have already been processed for sequence
``s``. Combined with the cached blocks, it determines the starting state for new tokens:
the most recent cached block before ``past_lens[s]`` is loaded, and the remaining
tokens up to ``past_lens[s]`` are replayed from that checkpoint.

**Attributes**

* *use_qk_l2norm*

  * **Description**: When ``True``, applies L2 normalization to query and key vectors before
    the recurrent update, using ``q_l2_norm_eps`` and ``k_l2_norm_eps`` as the minimum
    normalization denominators to avoid division by zero.
  * **Type**: ``boolean``
  * **Default value**: ``False``
  * **Required**: *no*

* *q_l2_norm_eps*

  * **Description**: Epsilon value used as the minimum denominator when L2-normalizing query
    vectors. Only used when ``use_qk_l2norm`` is ``True``.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: ``1e-6``
  * **Required**: *no*

* *k_l2_norm_eps*

  * **Description**: Epsilon value used as the minimum denominator when L2-normalizing key
    vectors. Only used when ``use_qk_l2norm`` is ``True``.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: ``1e-6``
  * **Required**: *no*

**Inputs**

* **0**: ``query`` - Tensor of type *T* and shape ``[batch_size_in_tokens, num_heads, key_head_dim]``.
  Query vectors for all tokens in the batch. **Required.**

* **1**: ``key`` - Tensor of type *T* and shape ``[batch_size_in_tokens, num_heads, key_head_dim]``.
  Key vectors for all tokens in the batch. **Required.**

* **2**: ``value`` - Tensor of type *T* and shape ``[batch_size_in_tokens, v_num_heads, value_head_dim]``.
  Value vectors for all tokens in the batch. **Required.**

* **3**: ``recurrent_state_table`` - Tensor of type *T* and shape
  ``[num_blocks, v_num_heads, value_head_dim, key_head_dim]``.
  Paged table of recurrent state snapshots. Each row is one block storing a complete state for
  all value heads at a cached token position. This tensor is updated in place during execution.
  The initial state before any tokens are processed is an all-zeros tensor. **Required.**

* **4**: ``gate`` - Tensor of type *T* and shape ``[batch_size_in_tokens, v_num_heads]``.
  Per-token, per-value-head gating scalar applied to the previous recurrent state in the delta
  update. **Required.**

* **5**: ``beta`` - Tensor of type *T* and shape ``[batch_size_in_tokens, v_num_heads]``.
  Per-token, per-value-head scalar controlling the magnitude of the delta update. **Required.**

* **6**: ``subsequence_begins`` - Tensor of type *T_IND* and shape ``[batch_size_in_sequences + 1]``.
  Start indices of each sequence's tokens in the flattened token batch (0-th dimension of
  ``query``, ``key``, ``value``). The tokens of sequence ``s`` span
  ``[subsequence_begins[s], subsequence_begins[s+1])``. **Required.**

* **7**: ``block_indices`` - Tensor of type *T_IND* and shape ``[num_blocks]``.
  Physical block row indices into ``recurrent_state_table``, concatenated across all sequences.
  For example, ``[0, 1, 3, 2, 4]`` with five blocks. **Required.**

* **8**: ``block_indices_begins`` - Tensor of type *T_IND* and shape
  ``[batch_size_in_sequences + 1]``.
  Splits ``block_indices`` among sequences. The block indices for sequence ``s`` are
  ``block_indices[block_indices_begins[s] : block_indices_begins[s+1]]``.
  For example, ``block_indices = [0, 1, 3, 2, 4]`` and ``block_indices_begins = [0, 3, 5]``
  means sequence 0 uses blocks ``[0, 1, 3]`` and sequence 1 uses blocks ``[2, 4]``. **Required.**

* **9**: ``past_lens`` - Tensor of type *T_IND* and shape ``[batch_size_in_sequences]``.
  Number of tokens already processed for each sequence. Used together with the cached states
  in ``recurrent_state_table`` to determine the starting recurrent state for each sequence.
  **Required.**

* **10**: ``cache_interval`` - Tensor of type *T_IND* and shape ``[batch_size_in_sequences]``.
  Interval (in tokens) at which the recurrent state is saved into a block of
  ``recurrent_state_table`` for each sequence. A value ``<= 0`` disables caching for that
  sequence. **Required.**

**Outputs**

* **0**: ``output_value`` - Tensor of type *T* and shape
  ``[batch_size_in_tokens, v_num_heads, value_head_dim]``.
  Per-token, per-value-head output vectors produced by querying the updated recurrent state.

.. note::

   ``recurrent_state_table`` (input 3) is updated in place as a side effect. The initial
   recurrent state is an all-zeros tensor, consistent with the StatefulCausalConv1D convention.

.. note::

   This operation uses grouped-query linear attention. The number of groups is
   ``num_groups = v_num_heads // num_heads``. Each query and key head is shared by
   ``num_groups`` consecutive value heads, with the mapping ``h_q = h_v // num_groups``.

**Types**

* *T*: any floating-point type.
* *T_IND*: ``int32`` or ``int64``.
