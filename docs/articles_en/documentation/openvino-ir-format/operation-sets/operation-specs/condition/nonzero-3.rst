NonZero
=======


.. meta::
  :description: Learn about NonZero-3 - an element-wise, condition operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *NonZero-3*

**Category**: *Condition*

**Short description**: *NonZero* returns the indices of the non-zero elements of the input tensor.

**Detailed description**: *NonZero* returns the indices of the non-zero elements of the input tensor (in row-major order - by dimension).

* The output tensor has shape ``[rank(input), num_non_zero]``.
* For example, for the tensor ``[[1, 0], [1, 1]]`` the output will be ``[[0, 1, 1], [0, 0, 1]]``.
* The output is a collection of tuples, each tuple has ``rank(input)`` elements and contains indices for a single non-zero element.
* The ``i``'th element of each output dimension is a part of ``i``'th tuple.
* In given example the tuples would be: ``[0, 0]``, ``[1, 0]``, ``[1, 1]``.

**Attributes**

* *output_type*

  * **Description**: the output tensor type
  * **Range of values**: ``i64`` or ``i32``
  * **Type**: string
  * **Default value**: "i64"
  * **Required**: *no*

**Inputs**:

*   **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**:

*   **1**: tensor with indices of non-zero elements of shape ``[rank(data), num_non_zero]`` of type *T_OUT*.

**Types**

* *T*: any type.

* *T_OUT*: Depending on *output_type* attribute can be ``int64`` or ``int32``.

**Example**

.. code-block:: xml
   :force:

    <layer ... type="NonZero">
        <data output_type="i64"/>
        <input>
            <port id="0">
                <dim>3</dim>
                <dim>10</dim>
                <dim>100</dim>
                <dim>200</dim>
            </port>
        </input>
        <output>
            <port id="1">
                <dim>4</dim>
                <dim>-1</dim>
            </port>
        </output>
    </layer>


