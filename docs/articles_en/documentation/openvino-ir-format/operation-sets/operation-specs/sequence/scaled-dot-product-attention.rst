ScaledDotProductAttention
=========================


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
	:force:

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

* **4**: ``attention_mask`` - two options available. ``attention_mask`` is ignored if ``causal`` is set to ``True``. **Optional.**

	* at least 2 dimensional tensor of type *T* or ``boolean`` and shape numpy-broadcastable to ``[N, ..., L, S]``. See :doc:`Numpy Broadcast Rules <../../broadcast-rules>` for broadcast details.

	* a scalar of type *T* with value ``0``. Scalar zero value signals that applying an attention mask is not necessary (similar to specifying attention_mask=None in the provided pseudo-code).

* **5**: ``scale`` a scalar or single element 1D tensor of type *T*, an alternative scale factor instead of 1/sqrt(query.shape[-1]) used by default in the pseudo-code above. **Optional.**


**Outputs**

* **1**: - the result of scaled dot-product attention, a tensor of type *T* and shape ``[N, ..., L, Ev]``.

**Types**

* *T*: any supported floating-point type.


**Dimensions**

* ``N, ...`` - one or more batch dimensions. Each batch dimension should be either constant across the input tensors (query, key, and value), indicating that they have the same batch size, or they should be numpy-broadcastable to the same value. See :doc:`Numpy Broadcast Rules <../../broadcast-rules>` for broadcast details.

* ``S`` - source sequence length

* ``L`` - target sequence length

* ``E`` - embedding dimension of the query and key

* ``Ev`` - embedding dimension of the value

At least one batch dimension ``N`` is required in ``query``, ``key`` and ``value`` inputs.
Other batch dimensions ``...`` are optional.


**Examples**

*Example 1: One batch dimension, dynamic dimensions support*

.. code-block:: xml
   :force:

    <layer id="285" name="aten::scaled_dot_product_attention_0" type="ScaledDotProductAttention" version="opset13">
			<data causal="false" />
			<input>
				<!-- Example with simple dimensions, with N = 1, L = -1, S = -1, E = 80, Ev = 80-->
				<port id="0" precision="FP32"> <!-- query -->
					<dim>1</dim> <!-- N -->
					<dim>-1</dim> <!-- L -->
					<dim>80</dim> <!-- E -->
				</port>
				<port id="1" precision="FP32"> <!-- key -->
					<dim>1</dim> <!-- N -->
					<dim>-1</dim> <!-- S -->
					<dim>80</dim> <!-- E -->
				</port>
				<port id="2" precision="FP32"> <!-- value -->
					<dim>1</dim> <!-- N -->
					<dim>-1</dim> <!-- S -->
					<dim>80</dim> <!-- Ev -->
				</port>
				<port id="3" precision="FP32"> <!-- attention_mask -->
					<dim>1</dim> <!-- N -->
					<dim>-1</dim> <!-- L -->
					<dim>-1</dim> <!-- S -->
				</port>
			</input>
			<output>
				<port id="4" precision="FP32">
					<dim>1</dim> <!-- N -->
					<dim>-1</dim> <!-- L -->
					<dim>80</dim> <!-- Ev -->
				</port>
			</output>
		</layer>

*Example 2: Matching multiple batch dimensions*

.. code-block:: xml
   :force:

    <layer id="286" name="aten::scaled_dot_product_attention_0" type="ScaledDotProductAttention" version="opset13">
			<data causal="false" />
			<input>
				<!-- Multiple batch dimensions: N1 = 1, N2 = 2, N3 = 3-->
				<port id="0" precision="FP32"> <!-- query -->
					<dim>1</dim> <!-- N1 -->
					<dim>2</dim> <!-- N2 -->
					<dim>3</dim> <!-- N3 -->
					<dim>-1</dim> <!-- L -->
					<dim>80</dim> <!-- E -->
				</port>
				<port id="1" precision="FP32"> <!-- key -->
					<dim>1</dim> <!-- N1 -->
					<dim>2</dim> <!-- N2 -->
					<dim>3</dim> <!-- N3 -->
					<dim>-1</dim> <!-- S -->
					<dim>80</dim> <!-- E -->
				</port>
				<port id="2" precision="FP32"> <!-- value -->
					<dim>1</dim> <!-- N1 -->
					<dim>2</dim> <!-- N2 -->
					<dim>3</dim> <!-- N3 -->
					<dim>-1</dim> <!-- S -->
					<dim>80</dim> <!-- Ev -->
				</port>
				<port id="3" precision="FP32"> <!-- attention_mask -->
					<dim>1</dim> <!-- N1 -->
					<dim>2</dim> <!-- N2 -->
					<dim>3</dim> <!-- N3 -->
					<dim>-1</dim> <!-- L -->
					<dim>-1</dim> <!-- S -->
				</port>
			</input>
			<output>
				<port id="4" precision="FP32">
					<dim>1</dim> <!-- N1 -->
					<dim>2</dim> <!-- N2 -->
					<dim>3</dim> <!-- N3 -->
					<dim>-1</dim> <!-- L -->
					<dim>80</dim> <!-- Ev -->
				</port>
			</output>
		</layer>

*Example 3: With batch dimensions broadcasting*

.. code-block:: xml
   :force:

    <layer id="287" name="aten::scaled_dot_product_attention_0" type="ScaledDotProductAttention" version="opset13">
			<data causal="false" />
			<input>
				<!-- Multiple batch dimensions, broadcastable to the following values: N1 = 4, N2 = 6, N3 = 10-->
				<port id="0" precision="FP32"> <!-- query -->
					<dim>4</dim> <!-- N1 (repeat 1 time) -->
					<dim>6</dim> <!-- N2 (repeat 1 time)-->
					<dim>10</dim> <!-- N3 (repeat 1 time)-->
					<dim>-1</dim> <!-- L -->
					<dim>80</dim> <!-- E -->
				</port>
				<port id="1" precision="FP32"> <!-- key -->
					<dim>1</dim> <!-- N1 (repeat 4 times) -->
					<dim>6</dim> <!-- N2 (repeat 1 time) -->
					<dim>10</dim> <!-- N3 (repeat 1 time) -->
					<dim>-1</dim> <!-- S -->
					<dim>80</dim> <!-- E -->
				</port>
				<port id="2" precision="FP32"> <!-- value -->
					<dim>1</dim> <!-- N1 (repeat 4 times)-->
					<dim>1</dim> <!-- N2 (repeat 6 times)-->
					<dim>1</dim> <!-- N3 (repeat 10 times)-->
					<dim>-1</dim> <!-- S -->
					<dim>80</dim> <!-- Ev -->
				</port>
				<port id="3" precision="FP32"> <!-- attention_mask -->
					<dim>1</dim> <!-- N1 (repeat 4 times)-->
					<dim>1</dim> <!-- N2 (repeat 6 times)-->
					<dim>1</dim> <!-- N3 (repeat 10 times)-->
					<dim>-1</dim> <!-- L -->
					<dim>-1</dim> <!-- S -->
				</port>
			</input>
			<output>
				<!-- Output contains broadcasted dimensions N1 = 4, N2 = 6, N3 = 10-->
				<port id="4" precision="FP32">
					<dim>4</dim> <!-- N1 -->
					<dim>6</dim> <!-- N2 -->
					<dim>10</dim> <!-- N3 -->
					<dim>-1</dim> <!-- L -->
					<dim>80</dim> <!-- Ev -->
				</port>
			</output>
		</layer>

*Example 5: With attention mask broadcasting*

.. code-block:: xml
   :force:

    <layer id="285" name="aten::scaled_dot_product_attention_0" type="ScaledDotProductAttention" version="opset13">
			<data causal="false" />
			<input>
				<!-- Example with simple dimensions, with N = 2, L = 16, S = 32, E = 80, Ev = 80-->
				<port id="0" precision="FP32"> <!-- query -->
					<dim>2</dim>  <!-- N -->
					<dim>16</dim> <!-- L -->
					<dim>80</dim> <!-- E -->
				</port>
				<port id="1" precision="FP32"> <!-- key -->
					<dim>2</dim>  <!-- N -->
					<dim>32</dim> <!-- S -->
					<dim>80</dim> <!-- E -->
				</port>
				<port id="2" precision="FP32"> <!-- value -->
					<dim>2</dim>  <!-- N -->
					<dim>32</dim> <!-- S -->
					<dim>80</dim> <!-- Ev -->
				</port>
				<port id="3" precision="FP32"> <!-- attention_mask -->
					<dim>2</dim>  <!-- N -->
					<dim>1</dim>  <!-- to be broadcasted to L -->
					<dim>1</dim> <!-- to be broadcasted to S -->
				</port>
			</input>
			<output>
				<port id="4" precision="FP32">
					<dim>2</dim>  <!-- N -->
					<dim>16</dim> <!-- L -->
					<dim>80</dim> <!-- Ev -->
				</port>
			</output>
		</layer>
