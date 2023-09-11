# BitwiseNot {#openvino_docs_ops_logical_BitwiseNot_13}

@sphinxdirective

.. meta::
  :description: Learn about BitwiseNot-13 - an element-wise, bitwise negation operation, which can be performed on a single input tensor.

**Versioned name**: *BitwiseNot-13*

**Category**: *Bitwise unary*

**Short description**: *BitwiseNot* performs bitwise logical negation operation with given tensor element-wise.

**Detailed description**: *BitwiseNot* performs bitwise logical negation operation for each element in given tensor, based on the following algorithm:

* Convert value from input tensor to binary representation according to used datatype,
* Perform logical negation on each bit in binary representation where `0` value is represents `false`, `1` value represents `true`,
* Convert binary representation according to input datatype.

Example 1 - *BitwiseNot* output for boolean Tenso:

.. code-block:: xml
   :force:

    <!-- For given boolean input: -->
    input = [True, False]
    <!-- Create binary representation of boolean: -->
    binary = [1, 0]
    <!-- Perform elementwise negation for each bit values: -->
    binary_bitwise_negation = [0, 1]
    <!-- Convert back binary values to boolean: -->
    bitwise_negation = [False, True]

    output = bitwise_negation

Example 2 - *BitwiseNot* output for uint8 tensor:

.. code-block:: xml
   :force:

    <!-- For given uint8 input: -->
    input = [1, 3] 
    <!-- Create binary representation of uint8: -->
    binary = [00000001, 00000011]
    <!-- Perform elementwise negation for each bit values: -->
    binary_bitwise_negation = [11111110, 11111100]
    <!-- Convert back binary values to uint8: -->
    bitwise_negation = [254, 252]

    output = bitwise_negation



**Attributes**: *BitwiseNot* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of bitwise logical negation operation. A tensor of type *T* and the same shape as input tensor.

**Types**

* *T*: ``any suported intiger or boolean type``.

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
