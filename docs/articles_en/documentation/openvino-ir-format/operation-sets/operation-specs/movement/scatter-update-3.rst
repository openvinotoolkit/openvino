ScatterUpdate
=============


.. meta::
  :description: Learn about ScatterUpdate-3 - a data movement operation, which can be
                performed on four required input tensors.

**Versioned name**: *ScatterUpdate-3*

**Category**: *Data movement*

**Short description**: *ScatterUpdate* creates a copy of the first input tensor with updated elements specified with second and third input tensors.

**Detailed description**: *ScatterUpdate* creates a copy of the first input tensor with updated elements in positions specified with ``indices`` input
and values specified with ``updates`` tensor starting from the dimension with index ``axis``. For the ``data`` tensor of shape :math:`[d_0,\;d_1,\;\dots,\;d_n]`, ``indices`` tensor of shape :math:`[i_0,\;i_1,\;\dots,\;i_k]` and ``updates`` tensor of shape :math:`[d_0,\;d_1,\;\dots,\;d_{axis - 1},\;i_0,\;i_1,\;\dots,\;i_k,\;d_{axis + 1},\;\dots, d_n]` the operation computes for each ``m, n, ..., p`` of the ``indices`` tensor indices:

.. math::

	data[\dots,\;indices[m,\;n,\;\dots,\;p],\;\dots] = updates[\dots,\;m,\;n,\;\dots,\;p,\;\dots]

where first :math:`\dots` in the ``data`` corresponds to :math:`[d_0,\;\dots,\;d_{axis - 1}]` dimensions, last :math:`\dots` in the ``data`` corresponds to the ``rank(data) - (axis + 1)`` dimensions.

Several examples for case when `axis = 0`:

1. ``indices`` is a :math:`0` D tensor: :math:`data[indices,\;\dots] = updates[\dots]`
2. ``indices`` is a :math:`1` D tensor (:math:`\forall_{i}`): :math:`data[indices[i],\;\dots] = updates[i,\;\dots]`
3. ``indices`` is a :math:`N` D tensor (:math:`\forall_{i,\;\dots,\;j}`): :math:`data[indices[i],\;\dots,\;j],\;\dots] = updates[i,\;\dots,\;j,\;\dots]`

**Attributes**: *ScatterUpdate* does not have attributes.

**Inputs**:

*   **1**: ``data`` tensor of arbitrary rank ``r`` and type *T_NUMERIC*. **Required.**

*   **2**: ``indices`` tensor with indices of type *T_IND*. All index values are expected to be within bounds ``[0, s - 1]`` along the axis
    of size ``s``. If multiple indices point to the same output location, the order of updating the values is undefined.
    If an index points to a non-existing output tensor element or is negative, then an exception is raised. **Required.**

*   **3**: ``updates`` tensor of type *T_NUMERIC* and rank equal to ``rank(indices) + rank(data) - 1`` **Required.**

*   **4**: ``axis`` tensor with scalar or 1D tensor with one element of type *T_AXIS* specifying axis for scatter.
    The value can be in the range ``[ -r, r - 1]``, where ``r`` is the rank of ``data``. **Required.**

**Outputs**:

*   **1**: tensor with shape equal to ``data`` tensor of the type *T_NUMERIC*.

**Types**

* *T_NUMERIC*: any numeric type.

* *T_IND*: any supported integer types.

* *T_AXIS*: any supported integer types.

**Examples**

*Example 1*

.. code-block:: xml
   :force:

    <layer ... type="ScatterUpdate">
        <input>
            <port id="0">  <!-- data -->
                <dim>1000</dim>
                <dim>256</dim>
                <dim>10</dim>
                <dim>15</dim>
            </port>
            <port id="1">  <!-- indices -->
                <dim>125</dim>
                <dim>20</dim>
            </port>
            <port id="2">  <!-- updates -->
                <dim>1000</dim>
                <dim>125</dim>
                <dim>20</dim>
                <dim>10</dim>
                <dim>15</dim>
            </port>
            <port id="3">   <!-- axis -->
                <dim>1</dim> <!-- value [1] -->
            </port>
        </input>
        <output>
            <port id="4" precision="FP32"> <!-- output -->
                <dim>1000</dim>
                <dim>256</dim>
                <dim>10</dim>
                <dim>15</dim>
            </port>
        </output>
    </layer>

*Example 2*

.. code-block:: xml
   :force:

    <layer ... type="ScatterUpdate">
        <input>
            <port id="0">  <!-- data -->
                <dim>3</dim>    <!-- {{-1.0f, 1.0f, -1.0f, 3.0f, 4.0f},  -->
                <dim>5</dim>    <!-- {-1.0f, 6.0f, -1.0f, 8.0f, 9.0f},   -->
            </port>             <!-- {-1.0f, 11.0f, 1.0f, 13.0f, 14.0f}} -->
            <port id="1">  <!-- indices -->
                <dim>2</dim> <!-- {0, 2} -->
            </port>
            <port id="2">  <!-- updates -->
                <dim>3</dim> <!-- {1.0f, 1.0f} -->
                <dim>2</dim> <!-- {1.0f, 1.0f} -->
            </port>          <!-- {1.0f, 2.0f} -->
            <port id="3">   <!-- axis -->
                <dim>1</dim> <!-- {1} -->
            </port>
        </input>
        <output>
            <port id="4">  <!-- output -->
                <dim>3</dim>    <!-- {{1.0f, 1.0f, 1.0f, 3.0f, 4.0f},   -->
                <dim>5</dim>    <!-- {1.0f, 6.0f, 1.0f, 8.0f, 9.0f},    -->
            </port>             <!-- {1.0f, 11.0f, 2.0f, 13.0f, 14.0f}} -->
        </output>
    </layer>



