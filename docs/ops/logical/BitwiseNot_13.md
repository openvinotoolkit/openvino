# BitwiseNot {#openvino_docs_ops_logical_BitwiseNot_13}

@sphinxdirective

.. meta::
  :description: Learn about BitwiseNot-13 - an element-wise, bitwise negation operation, which can be performed on a single input tensor.

**Versioned name**: *BitwiseNot-13*

**Category**: *Bitwise unary*

**Short description**: *BitwiseNot* performs bitwise logical negation operation with given tensor element-wise.

**Detailed description**: *BitwiseNot* performs bitwise logical negation operation for each element in given tensor, based on the following algorithm:

* Convert value from input tensor to binary representation according to input tensor datatype,
* Perform logical negation on each bit in binary representation, where `0` value is represents `false` and `1` value represents `true`,
* Convert back binary representation to input datatype.

Example 1 - *BitwiseNot* output for boolean tensor:

.. code-block:: xml
    :force:

    <!-- For given boolean input: -->
    input = [True, False]
    <!-- Create binary representation of boolean: -->
    <!-- [1, 0] -->
    <!-- Perform bitwise negation: -->
    <!-- [0, 1] -->
    <!-- Convert back binary values to boolean: -->
    output = [False, True]

Example 2 - *BitwiseNot* output for uint8 tensor:

.. code-block:: xml
    :force:

    <!-- For given uint8 input: -->
    input = [1, 3] 
    <!-- Create binary representation of uint8: -->
    <!-- [00000001, 00000011] -->
    <!-- Perform bitwise negation: -->
    <!-- [11111110, 11111100] -->
    <!-- Convert back binary values to uint8: -->
    output = [254, 252]

**Attributes**: *BitwiseNot* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of bitwise logical negation operation. A tensor of type *T* and the same shape as input tensor.

**Types**

* *T*: ``any suported integer or boolean type``.

**Example**

.. code-block:: xml
    :force:

    <layer ... type="BitwiseNot">
        <input>
            <port id="0">
                <dim>256</dim>
                <dim>56</dim>
            </port>
        </input>
        <output>
            <port id="1">
                <dim>256</dim>
                <dim>56</dim>
            </port>
        </output>
    </layer>


@endsphinxdirective
