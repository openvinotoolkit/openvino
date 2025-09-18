.. {#openvino_docs_ops_internal_MOE}
MOE
===


.. meta::
  :description: Learn about MOE - a basic block for the mixture of experts.

**Versioned name**: *MOE*

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

  * **Description**: The number of activated expert. Must be less than or equal to ``expert_num``.
  * **Range of values**: a positive integer number
  * **Type**: ``size_t``
  * **Required**: *yes*

* *expert_num*

  * **Description**: The number of expert number.
  * **Range of values**: a positive integer number
  * **Type**: ``size_t``
  * **Required**: *yes*

* *hidden_size*

  * **Description**: Feature size which is extracted from ``hidden_states``.
  * **Range of values**: a positive integer number
  * **Type**: ``size_t``
  * **Required**: *yes*

* *intermediate_size*

  * **Description**: Intermediate size which is extracted from expert_layer mentioned in the pseudo-code.
  * **Range of values**: a positive integer number
  * **Type**: ``size_t``
  * **Required**: *yes*

* *group_size*

  * **Description**: Weight compression group size which is extracted from expert_layer mentioned in the pseudo-code.
  * **Range of values**: a greater than or equal to 0 integer number
  * **Type**: ``size_t``
  * **Required**: *no*

* *weight_type*

  * **Description**: Weight data type which are extracted from expert_layer mentioned in the pseudo-code.
  * **Range of values**: "f16", "f32", "u8", "u4"
  * **Required**: *yes*

* *scale_type*

  * **Description**: Scale data type which are extracted from expert_layer mentioned in the pseudo-code.
  * **Range of values**: "f16", "dynamic"
  * **Required**: *no*

* *zp_type*

  * **Description**: Zero point data type which are extracted from expert_layer mentioned in the pseudo-code.
  * **Range of values**: "u8", "u4", "dynamic"
  * **Required**: *no*

* *gates/ups/downs*

  * **Description**: Weight data which are extracted from expert_layer mentioned in the pseudo-code.
  * **Type**: ``v0::Constant``
  * **Required**: *yes*

**Inputs**

* **1**: ``hidden_states`` - 2 dimensional tensor of type *T* with the shape [batch, hidden_size]. **Required.**

* **2**: ``router_logits`` - 2 dimensional tensor of type *T* with the shape [batch, expert_num]. **Required.**


**Outputs**

* **1**: Output tensor of the same shape and type as the ``hidden_states`` input tensor.

**Types**

* *T*: any floating point type.

**Example**

.. code-block:: xml
   :force:
		<layer id="5" name="moe_router" type="MOE" version="ie_internal_opset">
			<data config.topk="2" config.expert_num="4" config.hidden_size="2048" config.intermediate_size="768" config.group_size="128" config.fused_router_logic="1" config.weight_type="u4" config.scale_type="f16" config.zp_type="u4" expert0_mlp0.element_type="u4" expert0_mlp0.shape="768, 16, 128" expert0_mlp1.element_type="f16" expert0_mlp1.shape="768, 16, 1" expert0_mlp2.element_type="u4" expert0_mlp2.shape="768, 16, 1" expert1_mlp0.element_type="u4" expert1_mlp0.shape="768, 16, 128" expert1_mlp1.element_type="f16" expert1_mlp1.shape="768, 16, 1" expert1_mlp2.element_type="u4" expert1_mlp2.shape="768, 16, 1" expert2_mlp0.element_type="u4" expert2_mlp0.shape="768, 16, 128" expert2_mlp1.element_type="f16" expert2_mlp1.shape="768, 16, 1" expert2_mlp2.element_type="u4" expert2_mlp2.shape="768, 16, 1" expert3_mlp0.element_type="u4" expert3_mlp0.shape="768, 16, 128" expert3_mlp1.element_type="f16" expert3_mlp1.shape="768, 16, 1" expert3_mlp2.element_type="u4" expert3_mlp2.shape="768, 16, 1" />
			<input>
				<port id="0" precision="FP32">
					<dim>-1</dim>
					<dim>2048</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>-1</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>-1</dim>
					<dim>2048</dim>
				</port>
			</output>
		</layer>