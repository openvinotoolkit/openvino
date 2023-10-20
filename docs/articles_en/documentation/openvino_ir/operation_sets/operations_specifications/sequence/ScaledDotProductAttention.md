# ScaledDotProductAttention {#openvino_docs_ops_sequence_ScaledDotProductAttention_13}

@sphinxdirective

.. meta::
  :description: Learn about ScaledDotProductAttention-13 - a basic block for the transformer attention mechanism.

**Versioned name**: *ScaledDotProductAttention-13*

**Category**: *Sequence processing*

**Short description**: *ScaledDotProductAttention* partially implements
`torch.nn.functional.scaled_dot_product_attention <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>`__,
omitting training-related parameter.

**Detailed description**:

*ScaledDotProductAttention* provides functionality according to the following pseudo-code using other operations from OpenVINO opset and ``numpy``:

.. code-block:: py

def ScaledDotProductAttention(query, key, value, attn_mask=None, scale=None, *, causal):
    L, S = Gather(ShapeOf(query), -2), Gather(ShapeOf(key), -2)
    if scale is None:
        scale = 1.0 / Sqrt(ConvertLike(Gather(ShapeOf(query), -1), query))
    attn_bias = Broadcast(ConvertLike(0, query), [L, S])
    if causal:
        attn_bias = numpy.triu(Broadcast(ConvertLike(-inf, query), [L, S]), k=1)
    elif attn_mask is not None:
        if attn_mask.element_type == boolean:
            attn_bias = Select(LogicalNot(attn_mask), ConvertLike(-inf, query), ConvertLike(0, query))
        else:
            attn_bias += attn_mask
    attn_weight = MatMul(query, Transpose(key, [-2, -1])) * scale
    attn_weight += attn_bias
    attn_weight = Softmax(attn_weight, axis=-1)
    return MatMul(attn_weight, value)


**Attributes**

* *causal*

  * **Description**: If true, assumes causal attention masking according to the pseudo-code. In this case ``attention_mask`` input described below is ignored.
  * **Range of values**: a boolean value
  * **Type**: ``bool``
  * **Required**: *yes*


**Inputs**

* **1**: ``query`` - at least 3 dimensional tensor of type *T* and shape ``[N, ..., L, E]``. **Required.**

* **2**: ``key`` - at least 3 dimensional tensor of type *T* and shape ``[N, ..., S, E]``. **Required.**

* **3**: ``value`` - at least 3 dimensional tensor of type *T* and shape ``[N, ..., S, Ev]``. **Required.**

* **4**: ``attention_mask`` - two options:
	** at least 3 dimensional tensor of type *T* or ``boolean`` and shape ``[M, ..., L, S]``, or
	** a scalar of type *T* with value ``0``. Scalar zero value is used to indicate that `attention_mask` is really not required to be applied (``attention_mask=None`` in the pseudo-code above) but ``scale`` is required to be set.

	``attention_mask`` is ignored if ``causal`` is set to ``True``. **Optional.**

* **5**: ``scale`` a scalar tensor of type *T*, an alternative scale factor instead of 1/sqrt(query.shape[-1]) used by default in the pseudo-code above. **Optional.**


**Outputs**

* **1**: - the result of scaled dot-product attention, a tensor of type *T* and shape ``[N, ..., L, Ev]``.

**Types**

* *T*: any supported floating-point type.


**Dimensions**

* ``N, ...`` - one or more batch dimensions

* ``S`` - source sequence length

* ``L`` - target sequence length

* ``E`` - embedding dimension of the query and key

* ``Ev`` - embedding dimension of the value

* ``M, ...`` - one of more batch dimensions of the mask, should be broadcastable to ``N, ...``

At least one batch dimension ``N`` is required and should match among ``query``, ``key`` and ``value`` inputs.
Other batch dimensions ``...`` are optional, if present should match among ``query``, ``key`` and ``value`` inputs as well.


**Example**

.. code-block:: xml
   :force:

    <layer id="285" name="aten::scaled_dot_product_attention_0" type="ScaledDotProductAttention" version="opset13">
			<data causal="false" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>-1</dim>
					<dim>80</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>-1</dim>
					<dim>80</dim>
				</port>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>-1</dim>
					<dim>80</dim>
				</port>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>-1</dim>
					<dim>-1</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="FP32">
					<dim>1</dim>
					<dim>32</dim>
					<dim>-1</dim>
					<dim>80</dim>
				</port>
			</output>
		</layer>

@endsphinxdirective
