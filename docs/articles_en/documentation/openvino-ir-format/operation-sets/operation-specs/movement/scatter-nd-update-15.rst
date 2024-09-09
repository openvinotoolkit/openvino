ScatterNDUpdate
===============


.. meta::
  :description: Learn about ScatterNDUpdate-15 - a data movement operation, which can be
                performed on three required input tensors.

**Versioned name**: *ScatterNDUpdate-15*

**Category**: *Data movement*

**Short description**: Creates a copy of the ``data`` input tensor with updates for elements specified by ``indices`` by elements from ``updates`` according to *reduction* attribute.

**Detailed description**: The operation produces a copy of ``data`` tensor and updates its value using logic from ``reduction`` attribute, using values specified
by ``updates`` at specific index positions specified by ``indices``. The output shape is the same as the shape of ``data``.
If multiple indices point to the same output location then the order of updating the values is undefined.

The last dimension of ``indices`` corresponds to indices into elements if ``indices.shape[-1]`` = ``data.shape.rank`` or slices
if ``indices.shape[-1]`` < ``data.shape.rank``.
Input ``updates`` is a tensor with shape ``indices.shape[:-1] + data.shape[indices.shape[-1]:]``.

The operation to perform between the corresponding elements is specified by reduction attribute, by default the elements of data tensor are simply overwritten by the values from updates input.

Operator ScatterNDUpdate-15 is an equivalent to following NumPy snippet:

.. code-block:: py

    def scatter_nd_update_15(data, indices, updates, reduction=None):
        func = lambda x, y: y
        if reduction == "sum":
            func = lambda x, y: x + y
        elif reduction == "sub":
            func = lambda x, y: x - y
        elif reduction == "prod":
            func = lambda x, y: x * y
        elif reduction == "max":
            func = max
        elif reduction == "min":
            func = min
        out = np.copy(data)
        # Order of loop iteration is undefined.
        for ndidx in np.ndindex(indices.shape[:-1]):
            out[tuple(indices[ndidx])] = func(tuple(out[indices[ndidx]]), updates[ndidx])
        return out

Example 1 that shows simple case of update with *reduction* set to ``none``.:

.. code-block:: cpp

    data    = [1, 2, 3, 4, 5, 6, 7, 8]
    indices = [[4], [3], [1], [7], [-2], [-4]]
    updates = [9, 10, 11, 12, 13, 14]
    output  = [1, 11, 3, 10, 4, 6, 13, 12]


Example that shows update of two slices of ``4x4`` shape in ``data``, with *reduction* set to ``none``:

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
  * **Range of values**: one of ``none``, ``sum``, ``sub``, ``prod``, ``min``, ``max``
  * **Type**: `string`
  * **Default value**: ``none``
  * **Required**: *no*

**Inputs**:

*   **1**: ``data`` tensor of arbitrary rank ``r`` >= 1 and of type *T*. **Required.**

*   **2**: ``indices`` tensor with indices of arbitrary rank ``q`` >= 1 and of type *T_IND*. All index values ``i_j`` in index entry ``(i_0, i_1, ...,i_k)`` (where ``k = indices.shape[-1]``) must be within bounds ``[-s_j, s_j - 1]`` where ``s_j = data.shape[j]``. ``k`` must be at most ``r``. If multiple indices point to the same output location then the order of updating the values is undefined. Negative value of index means reverse indexing and will be normalized to value ``len(data.shape[j] + index)``. If an index points to non-existing element then exception is raised. **Required.**

*   **3**: ``updates`` tensor of rank ``r - indices.shape[-1] + q - 1`` of type *T*. If expected ``updates`` rank is 0D it can be a tensor with single element. **Required.**

**Outputs**:

*   **1**: tensor with shape equal to ``data`` tensor of the type *T*.

**Types**

* *T*: any numeric type. For boolean type, reduction sum, sub, prod behaves like logical OR, XOR, AND accordingly.

* *T_IND*: ``int32`` or ``int64``

**Example**

*Example 1*

.. code-block:: xml

    <layer ... reduction="none" type="ScatterNdUpdate">
        <input>
            <port id="0" precision="FP32">  <!-- data -->
                <dim>4</dim>  <!-- values: [1, 2, 3, 4] -->
            </port>
            <port id="1" precision="I32">  <!-- indices -->
                <dim>5</dim>  <!-- values: [0, 2, -3, -3, 0] -->
            </port>
            <port id="2" precision="FP32">  <!-- updates -->
                <dim>5</dim>  <!-- values: [10, 20, 30, 40, 50] -->
            </port>
        </input>
        <output>
            <port id="3" precision="FP32">
                <dim>4</dim>  <!-- values: [50, 20, 20, 4] -->
            </port>
        </output>
    </layer>

*Example 2*

.. code-block:: xml

    <layer ... reduction="sum" type="ScatterNdUpdate">
        <input>
            <port id="0" precision="FP16">  <!-- data -->
                <dim>4</dim>  <!-- values: [1, 2, 3, 4] -->
            </port>
            <port id="1" precision="I32">  <!-- indices -->
                <dim>5</dim>  <!-- values: [0, 2, -3, -3, 0] -->
            </port>
            <port id="2" precision="FP16">  <!-- updates -->
                <dim>5</dim>  <!-- values: [10, 20, 30, 40, 50] -->
            </port>
        </input>
        <output>
            <port id="3" precision="FP16">
                <dim>4</dim>  <!-- values: [61, 72, 23, 4] -->
            </port>
        </output>
    </layer>

*Example 3*

.. code-block:: xml

    <layer ... reduction="sub" type="ScatterNdUpdate">
        <input>
            <port id="0" precision="I32">  <!-- data -->
                <dim>4</dim>  <!-- values: [1, 2, 3, 4] -->
            </port>
            <port id="1" precision="I32">  <!-- indices -->
                <dim>5</dim>  <!-- values: [0, 2, -3, -3, 0] -->
            </port>
            <port id="2" precision="I32">  <!-- updates -->
                <dim>5</dim>  <!-- values: [10, 20, 30, 40, 50] -->
            </port>
        </input>
        <output>
            <port id="3" precision="I32">
                <dim>4</dim>  <!-- values: [-59, -68, -17, 4] -->
            </port>
        </output>
    </layer>

*Example 4*

.. code-block:: xml

    <layer ... reduction="prod" type="ScatterNdUpdate">
        <input>
            <port id="0" precision="FP32">  <!-- data -->
                <dim>4</dim>  <!-- values: [1, 2, 3, 4] -->
            </port>
            <port id="1" precision="I32">  <!-- indices -->
                <dim>5</dim>  <!-- values: [0, 2, -3, -3, 0] -->
            </port>
            <port id="2" precision="FP32">  <!-- updates -->
                <dim>5</dim>  <!-- values: [10, 20, 30, 40, 50] -->
            </port>
        </input>
        <output>
            <port id="3" precision="FP32">
                <dim>4</dim>  <!-- values: [500, 3600, 40, 4] -->
            </port>
        </output>
    </layer>

*Example 5*

.. code-block:: xml
   :force:

    <layer ... reduction="none" type="ScatterNDUpdate">
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
