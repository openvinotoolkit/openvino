SparseFillEmptyRows
======================


.. meta::
  :description: Learn about SparseFillEmptyRows-16 - a sparse operation, which
                can be performed on four required input tensors.

**Versioned name**: *SparseFillEmptyRows-16*

**Category**: *Sparse*

**Short description**: Fills empty rows of an input sparse tensor with a default value.

**Detailed description**:

Operation SparseFillEmptyRows is an implementation of ``tf.raw_ops.SparseFillEmptyRows`` for 2D sparse tensors only.

The input sparse tensor is represented by the three inputs: 

* ``indices``
* ``values``
* ``dense_shape``

For each row in the input 2D sparse tensor, this operator checks if the row is empty. If the row is empty, the operator adds an entry with the specified default value at index `[row, 0]`. The input may have empty columns at the end, which will not be affected by this operation.

The output sparse tensor will be in row-major order and will have the same dense shape as the `dense_shape` input, but with updated `output_indices` and `output_values`.

This operator also returns a boolean vector indicating which rows were filled with the default value: ``empty_row_indicator[i] = True`` if row ``i`` was an empty row.

**Attributes**: SparseFillEmptyRows-16 operation has no attributes.

**Inputs**:

* **1**: ``values`` 1D tensor containing the values of type *T* to be inserted at the specified indices. **Required.**
* **2**: ``dense_shape`` 1D tensor of type *T_IDX* indicating the shape of the 2D dense tensor. **Required.**
* **3**: ``indices`` 2D tensor of type *T_IDX* and non-negative values indicating the positions at which ``values`` are placed in the sparse tensor. **Required.**
    It is of shape ``[M, 2]``, where:

    * ``M`` is the same as the length of the ``values`` input.
    * The second dimension is always 2, as only 2D sparse tensors are supported.

* **4**: ``default_value`` a scalar of type *T* to be inserted into the empty rows. **Required.**

**Outputs**:

* **1**: ``output_indices`` 2D tensor of type *T_IDX* indicating the positions at which ``output_values`` are placed in the sparse tensor.
    It is of shape ``[M', 2]``, where:

    * ``M'`` is the length of the updated ``output_values``.
    * The second dimension is always 2, as only 2D sparse tensors are supported.

* **2**: ``output_values`` 1D tensor containing the values of type *T* to be inserted at the specified indices.
* **3**: ``empty_row_indicator`` 1D tensor of type ``boolean`` indicating True for rows which were empty before executing the operation.

**Types**

* *T*: any numeric type.
* *T_IDX*: ``int32`` or ``int64``.

**Example**

*Example 1: sparse tensor input with shape [5, 6].*

Input sparse tensor:

* ``indices = [[0, 1], [0, 3], [2, 0], [3, 1]]``
* ``values = [a, b, c, d]``
* ``dense_shape = [5, 6]``

Rows 1 and 4 are empty. The output sparse tensor will be:

* ``output_indices = [[0, 1], [0, 3], [1, 0], [2, 0], [3, 1], [4, 0]]``
* ``output_values = [a, b, default_value, c, d, default_value]``
* ``empty_row_indicator = [False, True, False, False, True]``

The output sparse tensor will be in row-major order and will have the same dense shape as the `dense_shape` input.

.. code-block:: xml

    <layer ... type="SparseFillEmptyRows" ... >
        <input>
            <port id="0" precision="FP32">      <!-- values are: [1, 3] -->
                <dim>2</dim>
            </port>
            <port id="1" precision="I32">       <!-- dense_shape value is: [3, 3] -->
                <dim>2</dim>
            </port>
            <port id="2" precision="I32">       <!-- indices value is: [[0, 0], [2, 2]] -->
                <dim>2</dim>
                <dim>2</dim>
            </port>
            <port id="3" precision="FP32">      <!-- default_value is: 42 -->
                <dim>0</dim>
            </port>
        </input>
        <output>
            <port id="4" precision="I32">       <!-- output_indices -->
                <dim>3</dim>
                <dim>2</dim>
            </port>
            <port id="5" precision="FP32">      <!-- output_values -->
                <dim>3</dim>
            </port>
            <port id="6" precision="BOOL">      <!-- empty_row_indicator -->
                <dim>3</dim>
            </port>
        </output>
    </layer>
