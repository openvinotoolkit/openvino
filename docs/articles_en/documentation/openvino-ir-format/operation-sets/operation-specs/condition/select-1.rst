Select
======


.. meta::
  :description: Learn about Select-1 - an element-wise, condition operation, which
                can be performed on three given tensors in OpenVINO.

**Versioned name**: *Select-1*

**Category**: *Condition*

**Short description**: *Select* returns a tensor filled with the elements from the second or the third inputs, depending on the condition (the first input) value.

**Detailed description**

*Select* takes elements from ``then`` input tensor or the ``else`` input tensor based on a condition mask provided in the first input ``cond``. Before performing selection, input tensors ``then`` and ``else`` are broadcasted to each other if their shapes are different and ``auto_broadcast`` attributes is not ``none``. Then the ``cond`` tensor is one-way broadcasted to the resulting shape of broadcasted ``then`` and ``else``. Broadcasting is performed according to ``auto_broadcast`` value.

**Attributes**

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input tensors.
  * **Range of values**:

    * *none* - no auto-broadcasting is allowed, all input shapes must match
    * *numpy* - numpy broadcasting rules, description is available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`
    * *pdpd* - PaddlePaddle-style implicit broadcasting, description is available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`
  * **Type**: ``string``
  * **Default value**: "numpy"
  * **Required**: *no*


**Inputs**:

* **1**: ``cond`` - tensor of type *T_COND* and arbitrary shape with selection mask. **Required**.

* **2**: ``then`` - tensor of type *T* and arbitrary shape with elements to take where the corresponding element in ``cond`` is ``true``. **Required**.

* **3**: ``else`` - tensor of type *T* and arbitrary shape with elements to take where the corresponding element in ``cond`` is ``false``. **Required**.


**Outputs**:

* **1**: blended output tensor that is tailored from values of inputs tensors ``then`` and ``else`` based on ``cond`` and broadcasting rules. It has the same type of elements as ``then`` and ``else``.

**Types**

* *T_COND*: ``boolean`` type.
* *T*: any supported numeric type.

**Example**

.. code-block:: xml
   :force:

    <layer ... type="Select">
        <input>
            <port id="0">     <!-- cond value is: [[false, false], [true, false], [true, true]] -->
                <dim>3</dim>
                <dim>2</dim>
            </port>
            <port id="1">     <!-- then value is: [[-1, 0], [1, 2], [3, 4]] -->
                <dim>3</dim>
                <dim>2</dim>
            </port>
            <port id="2">     <!-- else value is: [[11, 10], [9, 8], [7, 6]] -->
                <dim>3</dim>
                <dim>2</dim>
            </port>
        </input>
        <output>
            <port id="1">     <!-- output value is: [[11, 10], [1, 8], [3, 4]] -->
                <dim>3</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>



