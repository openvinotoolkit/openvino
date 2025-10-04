.. {#openvino_docs_ops_sequence_moe_16}

MOE
===


.. meta::
  :description: Learn about MOE - a basic block for the mixture of experts.

**Versioned name**: *MOE-16*

**Category**: *Sequence processing*

**Short description**: *MOE* partially implements
`Qwen3MoeSparseMoeBlock.forward <https://github.com/huggingface/transformers/blob/1fed6166c00b800330fcda8494f78cbcad8e4e3b/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L235-L263>`__,
omitting the `gate` operation.

**Detailed description**:

*MOE* provides functionality according to the following pseudo-code using torch:

.. code-block:: py
	:force:

	def MOE(hidden_states, router_logits, attrs):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, attrs.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=attrs.expert_num).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(attrs.expert_num):
            expert_layer = attrs.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states


**Attributes**

* *topk*

  * **Description**: The number of activated experts. Must be less than or equal to the number of experts.
  * **Range of values**: a positive integer number
  * **Type**: ``size_t``
  * **Required**: *yes*

* *expert_num*

  * **Description**: The number of expert number.
  * **Range of values**: a positive integer number
  * **Type**: ``size_t``
  * **Required**: *yes*

* *router_mode*

  * **Description**: Mode of operation for the router component in the MoE layer.
  * **Range of values**: ``normalized`` (normalize routing weights), ``sparse_mixer`` (use sparse mixer approach), ``standard`` (standard routing without normalization), ``none`` (use external routing weights)
  * **Type**: ``string``
  * **Required**: *yes*

* *expert_mode*

  * **Description**: Activation function to use in the expert components. Corresponds to ONNX MoE activation_type.
  * **Range of values**: ``Silu3GEMM``, ``SwiGLU2GEMM``
  * **Type**: ``string``
  * **Required**: *yes*

* *fused_experts*

  * **Description**: Whether experts are fused into a single computation.
  * **Range of values**: boolean value (true/false)
  * **Type**: ``bool``
  * **Required**: *yes*

* *expert_alpha*

  * **Description**: Alpha parameter used by some activation functions in expert computation. Required for certain activations like SwiGLU.
  * **Range of values**: any numeric value
  * **Type**: any numeric type
  * **Default value**: ``1.0``
  * **Required**: *no*

* *expert_beta*

  * **Description**: Beta parameter used by some activation functions in expert computation. Required for certain activations like SwiGLU.
  * **Range of values**: any numeric value
  * **Type**: any numeric type
  * **Default value**: ``0.0``
  * **Required**: *no*

* *expert_gamma*

  * **Description**: Gamma parameter used by some activation functions in expert computation. Additional configuration parameter for specific activation types.
  * **Range of values**: any numeric value
  * **Type**: any numeric type
  * **Default value**: ``1.0``
  * **Required**: *no*


**Inputs**

* **1**: ``hidden_states`` - 2 dimensional tensor of type *T* with the shape [batch, hidden_size]. **Required.**

* **2**: ``router_logits`` - 2 dimensional tensor of type *T* with the shape [batch, expert_num]. **Required.**

* **3**: ``fc1_experts_weights`` - 3D input tensor with shape (num_experts, inter_size, hidden_size). **Required.**

* **4**: ``fc1_experts_bias`` - 2D optional input tensor with shape (num_experts, inter_size). **Optional.**

* **5**: ``fc2_experts_weights`` - 3D input tensor with shape (num_experts, hidden_size, inter_size). **Optional.**

* **6**: ``fc2_experts_bias`` - 2D optional input tensor with shape (num_experts, hidden_size). **Optional.**

* **7**: ``fc3_experts_weights`` - 3D optional input tensor with shape (num_experts, inter_size, hidden_size). **Optional.**

* **8**: ``fc3_experts_bias`` - 2D optional input tensor with shape (num_experts, inter_size). **Optional.**


**Outputs**

* **1**: Output tensor of the same shape and type as the ``hidden_states`` input tensor.

**Types**

* *T*: any floating point type.

**Example**

.. code-block:: xml
   :force:

		<layer id="5" name="moe_router" type="MOE" version="opset16">
			<data topk="2" router_mode="normalized" expert_mode="Silu3GEMM" expert_alpha="1.0" expert_beta="0.0" expert_gamma="1.0" fused_experts="false" />
			<input>
				<port id="0" precision="FP32">
					<dim>-1</dim>
					<dim>2048</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>-1</dim>
					<dim>4</dim>
				</port>
				<port id="2" precision="FP32">
					<dim>4</dim>
					<dim>8192</dim>
					<dim>2048</dim>
				</port>
				<port id="3" precision="FP32">
					<dim>4</dim>
					<dim>8192</dim>
				</port>
				<port id="4" precision="FP32">
					<dim>4</dim>
					<dim>2048</dim>
					<dim>8192</dim>
				</port>
				<port id="5" precision="FP32">
					<dim>4</dim>
					<dim>2048</dim>
				</port>
			</input>
			<output>
				<port id="6" precision="FP32">
					<dim>-1</dim>
					<dim>2048</dim>
				</port>
			</output>
		</layer>
