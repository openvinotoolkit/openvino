Transpose
=========


.. meta::
  :description: Learn about Transpose-1 - a data movement operation, which can be
                performed on two required input tensors.

**Versioned name**: *Transpose-1*

**Category**: *Data movement*

**Short description**: *Transpose* operation reorders the input tensor dimensions.

**Detailed description**: *Transpose* operation reorders the input tensor dimensions. Source indexes and destination indexes are bound by the formula:

.. math::

   [output[i(order[0]), i(order[1]), ..., i(order[N-1])] = input[i(0), i(1), ..., i(N-1)]\\ \quad \textrm{where} \quad i(j) \quad\textrm{is in the range} \quad [0, (input.shape[j]-1)]


**Attributes**: *Transpose* operation has no attributes.

**Inputs**:

* **1**: ``arg`` - the tensor to be transposed. A tensor of type *T* and arbitrary shape. **Required.**
* **2**: ``input_order`` - the permutation to apply to the axes of the first input shape. A 1D tensor of ``n`` elements *T_AXIS* type and shape ``[n]``, where ``n`` is the rank of the first input or `0`. The tensor's value must contain every integer in the range ``[0, n-1]``, but if an empty tensor is specified (shape ``[0]``), then the axes will be inverted. **Required.**

**Outputs**:

*   **1**: A tensor of type *T* and transposed shape according to the rules specified above.

**Types**

* *T*: any supported type.
* *T_AXIS*: any integer type.


**Examples**

*Example 1*

.. code-block:: xml
   :force:

    <layer ... type="Transpose">
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>3</dim>
                <dim>4</dim>
            </port>
            <port id="1">
                <dim>3</dim>  <!-- [2, 0, 1] -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>4</dim>
                <dim>2</dim>
                <dim>3</dim>
            </port>
        </output>
    </layer>


*Example 2: input_order = empty 1D tensor of Shape[0]*

.. code-block:: xml
   :force:

    <layer ... type="Transpose">
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>3</dim>
                <dim>4</dim>
            </port>
            <port id="1">
                <dim>0</dim> <!-- input_order is an empty 1D tensor -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>4</dim>
                <dim>3</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>

