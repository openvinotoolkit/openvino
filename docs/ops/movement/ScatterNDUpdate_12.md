# ScatterNDUpdate {#openvino_docs_ops_movement_ScatterNDUpdate_12}

@sphinxdirective

**Versioned name**: *ScatterNDUpdate-12*

**Category**: *Data movement*

**Short description**: Creates a copy of the first input tensor with elements updated according to the logic specified by *reduction* attribute and provided input tensors.

**Detailed description**: The operation produces a copy of ``data`` tensor and updates the elements with values specified
by ``updates`` at specific index positions provided by ``indices`` input. The output shape is the same as the shape of ``data``.
The ``indices`` tensor can have duplicated entries, but in case the *reduction* is "none" the result is undefined.

The last dimension of ``indices`` corresponds to indices into elements if ``indices.shape[-1]`` = ``data.shape.rank`` or slices
if ``indices.shape[-1]`` < ``data.shape.rank``.
The ``updates`` input is a tensor of shape ``indices.shape[:-1] + data.shape[indices.shape[-1]:]``

Examples that shows update of four single elements in ``data``:

- Overwrite without additional operation, reduction = "none" (default)

.. code-block:: cpp

    data    = [0, 0, 0, 0, 0, 0, 0, 0]
    indices = [[0], [2], [4], [6], [-1]]
    updates = [10, 20, 30, 40, 50]
    output  = [10, 0, 20, 0, 30, 0, 40, 50]


    data    = [1, 1, 1, 1, 1, 1, 1, 1]
    indices = [[0], [7], [2], [5], [-3]]
    updates = [10, 20, 30, 40, 101]
    output  = [10, 1, 30, 1, 40, 101, 1, 20]


- Update by adding corresponding elements, reduction = "sum"

.. code-block:: cpp

    data    = [1, 1, 1, 1, 1, 1, 1, 1]
    indices = [[0], [7], [2], [7], [-3]]
    updates = [10, 20, 30, 40, 101]
    output  = [11, 1, 31, 1, 1, 102, 1, 61]


- Update by multiplication of the corresponding elements, reduction = "prod"

.. code-block:: cpp

    data    = [2, 2, 2, 2, 2, 2, 2, 2]
    indices = [[0], [7], [2], [7], [-3]]
    updates = [10, 20, 30, 40, 101]
    output  = [20, 2, 60, 2, 2, 202, 2, 1600]


- Update with minimum value of the corresponding elements, reduction = "min"

.. code-block:: cpp

    data    = [100, 20, 300, 400, 50, 600, 700, 800]
    indices = [[0], [0], [2], [4], [-1]]
    updates = [10, 1000, 30, 500, 80]
    output  = [10, 20, 30, 400, 50, 600, 700, 80]

- Update with maximum value of the corresponding elements, reduction = "max"

.. code-block:: cpp

    data    = [100, 20, 300, 400, 50, 600, 700, 800]
    indices = [[0], [0], [2], [4], [-1]]
    updates = [10, 1000, 30, 500, 80]
    output  = [1000, 20, 300, 400, 500, 600, 700, 800]


Example 2 shows update of two slices of ``4x4`` shape in ``data``, with reduction = "none":

.. code-block:: cpp

    data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
              [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
              [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
              [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
    indices = [[0], [2]]
    updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
              [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
    output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
              [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
              [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
              [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]


**Attributes**:

* *reduction*

  * **Description**: The type of operation to perform on the inputs.
  * **Range of values**: one of ``copy``, ``sum``, ``prod``, ``mean``, ``min``, ``max``
  * **Type**: `string`
  * **Default value**: ``copy``
  * **Required**: *no*


**Inputs**:

*   **1**: ``data`` tensor of arbitrary rank ``r`` >= 1 and of type *T*. **Required.**

*   **2**: ``indices`` tensor with indices of arbitrary rank ``q`` >= 1 and of type *T_IND*. All index values ``i_j`` in index entry ``(i_0, i_1, ...,i_k)`` (where ``k = indices.shape[-1]``) must be within bounds ``[-s_j, s_j - 1]`` where ``s_j = data.shape[j]``. ``k`` must be at most ``r``. **Required.**

*   **3**: ``updates`` tensor of rank ``r - indices.shape[-1] + q - 1`` of type *T*. If expected ``updates`` rank is 0D it can be a tensor with single element. **Required.**

**Outputs**:

*   **1**: tensor with shape equal to ``data`` tensor of the type *T*.

**Types**

* *T*: any numeric type.

* *T_IND*: ``int32`` or ``int64``

**Example**

.. code-block:: cpp

    <layer ... type="ScatterNDUpdate">
        <input>
            <port id="0">
                <dim>1000</dim>
                <dim>256</dim>
                <dim>10</dim>
                <dim>15</dim>
            </port>
            <port id="1">
                <dim>25</dim>
                <dim>125</dim>
                <dim>3</dim>
            </port>
            <port id="2">
                <dim>25</dim>
                <dim>125</dim>
                <dim>15</dim>
            </port>
        </input>
        <output>
            <port id="3">
                <dim>1000</dim>
                <dim>256</dim>
                <dim>10</dim>
                <dim>15</dim>
            </port>
        </output>
    </layer>

@endsphinxdirective
