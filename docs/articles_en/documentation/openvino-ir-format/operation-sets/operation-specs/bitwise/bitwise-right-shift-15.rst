BitwiseRightShift
=================


.. meta::
  :description: Learn about BitwiseRightShift-15 - an element-wise, bitwise right shift operation.

**Versioned name**: *BitwiseRightShift-15*

**Category**: *Bitwise binary*

**Short description**: *BitwiseRightShift* performs a bitwise right shift operation operation, applying multi-directional broadcast rules.

**Detailed description**: Before performing the operation, input tensors *a* and *b* are broadcasted if their shapes are different and the ``auto_broadcast`` attribute is not ``none``. Broadcasting is performed according to the ``auto_broadcast`` value.

After broadcasting input tensors *a* and *b*, *BitwiseRightShift* performs a bitwise right shift operation for each corresponding element in the given tensors, based on the following equation:

.. math::

   o_{i} = a_{i} \gg b_{i}


.. note::

    If the number of shifts is negative, or if it equals or exceeds the total number of bits in the type **T**, the behavior can be undefined or implementation-defined (depends on the hardware).

    Unsigned integer shift is always performed modulo 2^n where n is the number of bits in the type **T**.

    When signed integer shift operation overflows (the result does not fit in the result type), the behavior is undefined.

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

* **1**: A tensor of type *T* and arbitrary shape, containing data to be shifted. **Required.**
* **2**: A tensor of type *T* and arbitrary shape, with the number of shifts.  **Required.**

**Outputs**

* **1**: The result of element-wise *BitwiseRightShift* operation. A tensor of type *T* and the shape equal to the broadcasted shape of two inputs.

**Types**

* *T*: ``any supported integer type``.

**Examples**

*Example 1: no broadcast*

.. code-block:: xml
    :force:

    <layer ... type="BitwiseRightShift">
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

    <layer ... type="BitwiseRightShift">
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
