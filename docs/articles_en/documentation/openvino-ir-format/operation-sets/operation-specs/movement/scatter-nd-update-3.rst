ScatterNDUpdate
===============


.. meta::
  :description: Learn about ScatterNDUpdate-3 - a data movement operation, which can be
                performed on three required input tensors.

**Versioned name**: *ScatterNDUpdate-3*

**Category**: *Data movement*

**Short description**: Creates a copy of the first input tensor with updated elements specified with second and third input tensors.

**Detailed description**: The operation produces a copy of ``data`` tensor and updates its value to values specified
by ``updates`` at specific index positions specified by ``indices``. The output shape is the same as the shape of ``data``.
``indices`` tensor must not have duplicate entries. In case of duplicate entries in ``indices`` the result is undefined.

The last dimension of ``indices`` can be at most the rank of ``data.shape``.
The last dimension of ``indices`` corresponds to indices into elements if ``indices.shape[-1]`` = ``data.shape.rank`` or slices
if ``indices.shape[-1]`` < ``data.shape.rank``. ``updates`` is a tensor with shape ``indices.shape[:-1] + data.shape[indices.shape[-1]:]``

Example 1 that shows update of four single elements in ``data``:

.. code-block:: cpp

    data    = [1, 2, 3, 4, 5, 6, 7, 8]
    indices = [[4], [3], [1], [7]]
    updates = [9, 10, 11, 12]
    output  = [1, 11, 3, 10, 9, 6, 7, 12]


Example 2 that shows update of two slices of ``4x4`` shape in ``data``:

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



**Attributes**: *ScatterNDUpdate* does not have attributes.

**Inputs**:

*   **1**: ``data`` tensor of arbitrary rank ``r`` >= 1 and of type *T*. **Required.**

*   **2**: ``indices`` tensor with indices of arbitrary rank ``q`` >= 1 and of type *T_IND*. All index values ``i_j`` in index entry ``(i_0, i_1, ...,i_k)`` (where ``k = indices.shape[-1]``) must be within bounds ``[0, s_j - 1]`` where ``s_j = data.shape[j]``. ``k`` must be at most ``r``. **Required.**

*   **3**: ``updates`` tensor of rank ``r - indices.shape[-1] + q - 1`` of type *T*. If expected ``updates`` rank is 0D it can be a tensor with single element. **Required.**

**Outputs**:

*   **1**: tensor with shape equal to ``data`` tensor of the type *T*.

**Types**

* *T*: any numeric type.

* *T_IND*: ``int32`` or ``int64``

**Example**

.. code-block:: xml
   :force:

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


