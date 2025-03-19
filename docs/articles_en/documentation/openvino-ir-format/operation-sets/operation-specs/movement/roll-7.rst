Roll
====


.. meta::
  :description: Learn about Roll-7 - a data movement operation, which can be
                performed on three required input tensors.

**Versioned name**: *Roll-7*

**Category**: *Data movement*

**Short description**: The *Roll* operation shifts elements of a tensor along specified axes.

**Detailed description**: *Roll* produces a tensor with the same shape as the first input tensor and with elements shifted along dimensions specified in the *axes* tensor. The shift size is specified in the *shift* input tensor. Elements that are shifted beyond the last position will be added in the same order starting from the first position.

Example 1. *Roll* output with ``shift`` = 1, ``axes`` = 0:

.. code-block:: cpp

    data    = [[ 1,  2,  3],
            [ 4,  5,  6],
            [ 7,  8,  9],
            [10, 11, 12]]
    output  = [[10, 11, 12],
            [ 1,  2,  3],
            [ 4,  5,  6],
            [ 7,  8,  9]]


Example 2. *Roll* output with ``shift`` = [-1, 2], ``axes`` = [0, 1]:

.. code-block:: cpp

    data    = [[ 1,  2,  3],
            [ 4,  5,  6],
            [ 7,  8,  9],
            [10, 11, 12]]
    output  = [[ 5,  6,  4],
            [ 8,  9,  7],
            [11, 12, 10],
            [ 2,  3,  1]]


Example 3. *Roll* output with ``shift`` = [1, 2, 1], ``axes`` = [0, 1, 0]:

.. code-block:: cpp

    data    = [[ 1,  2,  3],
            [ 4,  5,  6],
            [ 7,  8,  9],
            [10, 11, 12]]
    output  = [[ 8,  9,  7],
            [11, 12, 10],
            [ 2,  3,  1],
            [ 5,  6,  4]]


**Attributes**

No attributes available.

**Inputs**:

*   **1**: ``data`` a tensor of type *T*. **Required.**

*   **2**: a ``shift`` scalar or 1D tensor of type *T_IND_1*. Specifies the number of places by which the elements of the ``data`` tensor are shifted. If ``shift`` is a scalar, each dimension specified in the ``axes`` tensor are rolled by the same ``shift`` value. If ``shift`` is a 1D tensor, ``axes`` must be a 1D tensor of the same size, and each dimension from ``axes`` tensor are rolled by the corresponding value from the ``shift`` tensor. If the value of ``shift`` is positive, elements are shifted positively (towards larger indices). Otherwise, elements are shifted negatively (towards smaller indices). **Required.**

*   **3**: ``axes`` a scalar or 1D tensor of type *T_IND_2*. Specifies axes along which elements are shifted. If the same axis is referenced more than once, the total shift for that axis will be the sum of all the shifts that belong to that axis. If ``axes`` has negative value, axis index will be calculated using the formula: ``N_dims + axis``, where ``N_dims`` - total number of dimensions in the ``data`` tensor, ``axis`` - negative axis index from the ``axes`` tensor. **Required.**


**Outputs**:

*   **1**: output tensor with shape and type equal to the ``data`` tensor.

**Types**

* *T*: any supported type.
* *T_IND_1*: ``int32`` or ``int64``.
* *T_IND_2*: ``int32`` or ``int64``.

**Example**

*Example 1: "shift" and "axes" are 1D tensors.*

.. code-block:: xml
   :force:

    <layer ... type="Roll">
        <input>
            <port id="0">
                <dim>3</dim>
                <dim>10</dim>
                <dim>100</dim>
                <dim>200</dim>
            </port>
            <port id="1">
                <dim>2</dim>
            </port>
            <port id="2">
                <dim>2</dim> <!-- shifting along specified axes with the corresponding shift values -->
            </port>
        </input>
        <output>
            <port id="0">
                <dim>3</dim>
                <dim>10</dim>
                <dim>100</dim>
                <dim>200</dim>
            </port>
        </output>
    </layer>


*Example 2: "shift" value is a scalar and multiple axes are specified.*

.. code-block:: xml
   :force:

    <layer ... type="Roll">
        <input>
            <port id="0">
                <dim>3</dim>
                <dim>10</dim>
                <dim>100</dim>
                <dim>200</dim>
            </port>
            <port id="1">
                <dim>1</dim>
            </port>
            <port id="2">
                <dim>2</dim> <!-- shifting along specified axes with the same shift value -->
            </port>
        </input>
        <output>
            <port id="0">
                <dim>3</dim>
                <dim>10</dim>
                <dim>100</dim>
                <dim>200</dim>
            </port>
        </output>
    </layer>


