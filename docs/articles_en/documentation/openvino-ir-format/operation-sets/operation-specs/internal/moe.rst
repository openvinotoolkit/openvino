.. {#openvino_docs_ops_internal_MOE}

MOE
===

.. meta::
  :description: Learn about MOE - a Mixture of Experts block, receiving routing weights and active experts indices as inputs, and performing expert computation according to the selected expert_type.

**Versioned name**: *MOE*

**Category**: *Internal*

**Short description**:  
The *MOE* (Mixture of Experts) operation fuses the computation of multiple experts, using routing weights and indices to select and combine expert outputs.

**Detailed description**:  
The MOE op receives hidden states, routing weights, and indices of selected experts, along with expert weights and (optionally) biases. It performs the expert computation as specified by the `expert_type` attribute, applying the routing_weights and combining the results. This enables efficient, fused computation of Mixture of Experts architectures excluding the router part (computation of routing weights).

**Pseudocode for expert_type**
The ``router_topk_output_indices`` are used to select the top-k experts for optimized computation, not included in the pseudocode below.

* ``GEMM2_BIAS_SWIGLU_CLAMP``:
  .. code-block:: python
    # Common part: Reshape hidden states and prepare for expert computation
    reshaped_hidden_states = reshape(hidden_states, [-1, 0], special_zero=True)
    tiled_hidden_states = tile(reshaped_hidden_states, [num_experts, 1])
    reshaped_hidden_states = reshape(tiled_hidden_states, [num_experts, -1, 0], special_zero=True)

    # Experts computation part (GEMM2_BIAS_SWIGLU_CLAMP)
    # Fused gate_up computation
    gate_up = matmul(reshaped_hidden_states, weight_0, transpose_a=False, transpose_b=False) + bias_0
    # Slice gate_up into two halves along last dimension, taking every second element with step two
    slice_1 = gate_up[..., ::2]      # every second element starting from index 0
    slice_2 = gate_up[..., 1::2]     # every second element starting from index 1
    # Branch 1: Minimum and Swish
    minimum_1 = minimum(slice_2, expert_beta)
    swish_1 = swish(minimum_1, beta=expert_alpha)
    # Branch 2: Clamp and Add
    clamp_1 = clamp(slice_1, -expert_beta, expert_beta)
    add_1 = clamp_1 + 1
    # Multiply branches
    fused = add_1 * swish_1
    # Down projection
    down_proj = matmul(fused, weight_1, transpose_a=False, transpose_b=False) + bias_1

    # Common part: Routing and summation
    routed_experts = reshape(down_proj, [num_experts, batch_size, -1, hidden_size]) * routing_weights
    output = reduce_sum(routed_experts, axis=0, keep_dims=False)

* ``GEMM3_SWIGLU``:
  .. code-block:: python
    # Common part: Reshape hidden states and prepare for expert computation
    reshaped_hidden_states = reshape(hidden_states, [-1, 0], special_zero=True)
    tiled_hidden_states = tile(reshaped_hidden_states, [num_experts, 1])
    reshaped_hidden_states = reshape(tiled_hidden_states, [num_experts, -1, 0], special_zero=True)

    # Experts computation part (GEMM3_SWIGLU)
    x_proj = matmul(reshaped_hidden_states, weight_0, transpose_a=False, transpose_b=True)
    x_proj2 = matmul(reshaped_hidden_states, weight_1, transpose_a=False, transpose_b=True)
    swiglu = swish(x_proj, beta=expert_alpha)
    x_proj = x_proj * swiglu
    down_proj = matmul(swiglu, weight_2, transpose_a=False, transpose_b=True)
    
    # Common part: Routing and summation
    routed_experts = reshape(down_proj, [num_experts, batch_size, -1, hidden_size]) * routing_weights
    output = reduce_sum(routed_experts, axis=0, keep_dims=False)


**Attributes**

* *expert_type*

  * **Description**: Specifies the computation performed by each expert. Determines the sequence of operations (e.g., GEMM, activation, bias, clamp).
  * **Type**: ``enum`` (see below)
  * **Required**: *yes*
  * **Supported values**:
    * ``GEMM2_BIAS_SWIGLU_CLAMP``: Two GEMMs with bias, SwiGLU activation, and clamp.
    * ``GEMM3_SWIGLU``: Three GEMMs with SwiGLU activation.

* *expert_alpha*

  * **Description**: Alpha attribute for activation functions (used for Swish with GEMM2_BIAS_SWIGLU_CLAMP).
  * **Type**: ``float``
  * **Default value**: ``1.0``
  * **Required**: *no*

* *expert_beta*

  * **Description**: Beta attribute - used as value for clamp min/max bounds (used with GEMM2_BIAS_SWIGLU_CLAMP).
  * **Type**: ``float``
  * **Default value**: ``0.0``
  * **Required**: *no*

**Inputs**

* **0**: ``hidden_states``  
  *2D tensor* of type *T* with shape ``[batch, ..., hidden_size]``.  
  The input hidden representations.

* **1**: ``routing_weights``  
  *Tensor* of type *T* with shape ``[..., topk, 1]`` for example ``[num_experts, batch, topk, 1]``.  
  The normalized weights for the selected top-k experts (after routing/normalization).

* **2**: ``router_topk_output_indices``  
  *Tensor* of type *T_ind* with shape ``[..., topk]`` for example ``[batch, topk]``.  
  Indices of the selected top-k ("active") experts for each input.

* **3**: ``weight_0``  
  *Tensor* of type *T* with shape ``[num_experts, hidden_size, inter_size]``  
  or ``[num_experts, hidden_size, 2 * inter_size]`` if fused (e.g. with expert_type ``GEMM2_BIAS_SWIGLU_CLAMP``).  
  Weights for the first MatMul.

* **4**: ``bias_0`` *(required only for GEMM2_BIAS_SWIGLU_CLAMP)*  
  *Tensor* of type *T* with shape ``[num_experts, ...]`` broadcastable to the output of the first MatMul, for example ``[num_experts, 1, 2 * inter_size]`` if fused (e.g. with expert_type ``GEMM2_BIAS_SWIGLU_CLAMP``) or empty tensor.  
  Bias to be added after the first MatMul.

* **5**: ``weight_1``  
  *Tensor* of type *T* with shape ``[num_experts, inter_size, hidden_size]``.  
  Weights for the second MatMul.

* **6**: ``bias_1`` *(optional)*  
 *Tensor* of type *T* with shape ``[num_experts, ...]`` broadcastable to the output of the second MatMul or empty tensor.  
  Bias to be added after the second MatMul.

* **7**: ``weight_2`` *(optional)*  
  *Tensor* of type *T* with shape ``[num_experts, hidden_size, inter_size]``.  
  Weights for the third MatMul.

* **8**: ``bias_2`` *(optional, currently not used with any of the supported expert_types)*
  *Tensor* of type *T* with shape ``[num_experts, ...]`` broadcastable to the output of the second MatMul or empty tensor.   
  Bias to be added after the third MatMul.

.. note::

    Bias inputs are optional and can be omitted if no bias is used, for example with ``GEMM3_SWIGLU`` expert_type. Then the number of the weights should match the number of GEMMs.

**Outputs**

* **0**: Output tensor of type *T* with the same shape as hidden_states input ``[batch, ..., hidden_size]``.  
  The fused output of the selected experts, weighted by routing weights.

**Types**

* *T*: any floating point type.
* *T_ind*: INT64 or INT32.
