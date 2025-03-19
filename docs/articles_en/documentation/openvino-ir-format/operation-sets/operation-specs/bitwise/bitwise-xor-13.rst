BitwiseXor
==========


.. meta::
  :description: Learn about BitwiseXor-13 - an element-wise, bitwise XOR operation, which can be performed on two required input tensors.

**Versioned name**: *BitwiseXor-13*

**Category**: *Bitwise binary*

**Short description**: *BitwiseXor* performs a bitwise logical XOR operation with two given tensors element-wise, applying multi-directional broadcast rules.

**Detailed description**: Before performing the operation, input tensors *a* and *b* are broadcasted if their shapes are different and the ``auto_broadcast`` attribute is not ``none``. Broadcasting is performed according to the ``auto_broadcast`` value.

After broadcasting input tensors *a* and *b*, *BitwiseXor* performs a bitwise logical XOR operation for each corresponding element in the given tensors, based on the following algorithm.

For ``boolean`` type tensors, BitwiseXor is equivalent to :doc:`LogicalXor <../logical/logical-xor-1>`.

If tensor is of ``any supported integer`` type, for each element of the tensor:

1.  Convert values from input tensors to their binary representation according to the input tensor datatype.
2.  Perform a logical XOR on each bit in the binary representation of values from *a* and *b*, where value ``0`` represents ``false`` and value ``1`` represents ``true``.
3.  Convert the results of XOR in binary representation to the input datatype.

Example 1 - *BitwiseXor* output for boolean tensor:

.. code-block:: py
    :force:

    # For given boolean inputs:
    a = [True, False, False]
    b = [True, True, False]
    # Perform logical XOR operation same as in LogicalXor operator:
    output = [False, True, False]

Example 2 - *BitwiseXor* output for uint8 tensor:

.. code-block:: py
    :force:

    # For given uint8 inputs:
    a = [21, 120]
    b = [3, 37]
    # Create a binary representation of uint8:
    # binary a: [00010101, 01111000]
    # binary b: [00000011, 00100101]
    # Perform bitwise XOR of corresponding elements in a and b:
    # [00010110, 01011101]
    # Convert binary values to uint8:
    output = [22, 93]

**Attributes**:

* *auto_broadcast*

  * **Description**: specifies the rules used for auto-broadcasting of input tensors.
  * **Range of values**:

    * *none* - no auto-broadcasting is allowed, all input shapes must match,
    * *numpy* - numpy broadcasting rules, description is available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`,
    * *pdpd* - PaddlePaddle-style implicit broadcasting, description is available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`.

  * **Type**: string
  * **Default value**: "numpy"
  * **Required**: *no*

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**
* **2**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise *BitwiseXor* operation. A tensor of type *T* and the same shape equal to the broadcasted shape of two inputs.

**Types**

* *T*: ``any supported integer or boolean type``.

**Examples**

*Example 1: no broadcast*

.. code-block:: xml
    :force:

    <layer ... type="BitwiseXor">
        <input>
            <port id="0">
                <dim>256</dim>
                <dim>56</dim>
            </port>
            <port id="1">
                <dim>256</dim>
                <dim>56</dim>
            </port>
        </input>
        <output>
            <port id="2">
                <dim>256</dim>
                <dim>56</dim>
            </port>
        </output>
    </layer>


*Example 2: numpy broadcast*

.. code-block:: xml
    :force:

    <layer ... type="BitwiseXor">
        <input>
            <port id="0">
                <dim>8</dim>
                <dim>1</dim>
                <dim>6</dim>
                <dim>1</dim>
            </port>
            <port id="1">
                <dim>7</dim>
                <dim>1</dim>
                <dim>5</dim>
            </port>
        </input>
        <output>
            <port id="2">
                <dim>8</dim>
                <dim>7</dim>
                <dim>6</dim>
                <dim>5</dim>
            </port>
        </output>
    </layer>


