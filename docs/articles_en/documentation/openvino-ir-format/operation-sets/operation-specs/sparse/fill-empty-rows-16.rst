FillEmptyRows
======================


.. meta::
  :description: Learn about FillEmptyRows-16 - a sparse operation, which
                can be performed on four required and two optional input tensors, depending
                on the type of data provided to the op.

**Versioned name**: *FillEmptyRows-16*

**Category**: *Sparse*

**Short description**: Fills empty rows of an input sparse tensor with a default value.

**Detailed description**:

Operation FillEmptyRows is an implementation of ``tf.sparse.fill_empty_rows``.

For each row in the input sparse tensor, this operator checks if the row is empty. If the row is empty, the operator adds an entry with the specified default value at index [row, 0]. The input may have empty columns at the end, which will not be affected by this operation.

The output sparse tensor will be in row-major order and will have the same shape as the input.

This operator also returns an boolean vector indicating which rows were filled with the default value: ``empty_row_indicator[i] = True`` if row ``i`` was an empty row.

The operation can be performed on four required and two optional input tensors, depending on the type of data provided to the op:

* **If ``values`` is of numeric type** the ``begins`` and ``ends`` inputs are ignored. The operator has 4 outputs.
* **If ``values`` is of bytes type** the ``begins`` and ``ends`` inputs are required and they, along with ``values``, should be in form of outputs of a ``StringTensorUnpack`` operator. The operator has 6 outputs.

**Attributes**: FillEmptyRows-16 operation has no attributes.

**Inputs**:

* **1**: ``default_value`` a scalar of type *T* to be inserted into the empty rows. **Required.**
* **2**: ``values`` 1D tensor containing the values of type *T* to be inserted at the specified indices. **Required.**
* **3**: ``dense_shape`` 1D tensor of type *T_IDX* indicating the shape of the dense tensor. **Required.**
* **4**: ``indices`` 2D tensor of type *T_IDX* indicating the positions at which ``values`` are placed in the sparse tensor. **Required.**
    It is of shape ``[M, N]``, where:

    * ``M`` is the same as length of the ``values`` input.
    * ``N`` is equal to the rank of ``dense_shape``.
* **5**: ``begins`` ND tensor of non-negative integer numbers of type *T_IDX*, containing indices of each string's beginnings. **Required if ``values`` is of type bytes.**
* **6**: ``ends`` ND tensor of non-negative integer numbers of type *T_IDX*, containing indices of each string's endings. **Required if ``values`` is of type bytes.**

**Outputs**:

* **1**: ``values`` 1D tensor containing the values of type *T* to be inserted at the specified indices.
* **2**: ``dense_shape`` 1D tensor of type *T_IDX* indicating the shape of the dense tensor.
* **3**: ``indices`` 2D tensor of type *T_IDX* indicating the positions at which ``values`` are placed in the sparse tensor.
    It is of shape ``[M, N]``, where:

    * ``M`` is the same as length of the ``values`` input.
    * ``N`` is equal to the rank of ``dense_shape``.
* **4**: ``empty_row_indicator`` 1D tensor of type ``boolean`` indicating True for rows which were empty before executing the operation.
* **5**: ``begins`` tensor of non-negative integer numbers of type *T_IDX*, containing new indices of each string's beginnings. **Only if ``values`` input is of bytes type.**
* **6**: ``ends`` tensor of non-negative integer numbers of type *T_IDX*, containing new indices of each string's ends. **Only if ``values`` input is of bytes type.**

**Types**

* *T*: any numeric type or bytes.
* *T_IDX*: ``int32`` or ``int64``.

**Example**

*Example 1: ``values`` is of type ``FP32``.*

.. code-block:: xml

    <layer ... type="FillEmptyRows" ... >
        <input>
            <port id="0" precision="FP32">      <!-- default_value is: 42 -->
                <dim>0</dim>
            </port>
            <port id="1" precision="FP32">      <!-- values are: [1, 3] -->
                <dim>2</dim>
            </port>
            <port id="2" precision="I32">       <!-- dense_shape value is: [3, 3] -->
                <dim>2</dim>
            </port>
            <port id="3" precision="I32">       <!-- indices value is: [[0, 0], [2, 2]] -->
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </input>
        <output>
            <port id="4" precision="FP32">      <!-- values are: [1, 42, 3] -->
                <dim>3</dim>
            </port>
            <port id="5" precision="I32">       <!-- dense_shape value is: [3, 3] -->
                <dim>2</dim>
            </port>
            <port id="6" precision="I32">       <!-- indices value is: [[0, 0], [1, 0], [2, 2]] -->
                <dim>3</dim>
                <dim>2</dim>
            </port>
            <port id="7" precision="BOOL">   <!-- output value is: [False, True, False] -->
                <dim>3</dim>
            </port>
        </output>
    </layer>

*Example 2: ``values`` is of type ``U8``.* TODO: figure out how to represent sparse unpacked strings
