ScatterElementsUpdate
=====================


.. meta::
  :description: Learn about ScatterElementsUpdate-3 - a data movement operation, which can be
                performed on four required input tensors.

**Versioned name**: *ScatterElementsUpdate-3*

**Category**: *Data movement*

**Short description**: Creates a copy of the first input tensor with updated elements specified with second and third input tensors.

**Detailed description**: For each entry in ``updates``, the target index in ``data`` is obtained by combining the corresponding entry in
``indices`` with the index of the entry itself: the index-value for dimension equal to ``axis`` is obtained from the value of the corresponding entry in
``indices`` and the index-value for dimension not equal to ``axis`` is obtained from the index of the entry itself.

For instance, in a 3D tensor case, the update corresponding to the ``[i][j][k]`` entry is performed as below:

.. code-block:: cpp

    output[indices[i][j][k]][j][k] = updates[i][j][k] if axis = 0,
    output[i][indices[i][j][k]][k] = updates[i][j][k] if axis = 1,
    output[i][j][indices[i][j][k]] = updates[i][j][k] if axis = 2


``update`` tensor dimensions are less or equal to the corresponding ``data`` tensor dimensions.

**Attributes**: *ScatterElementsUpdate* does not have attributes.

**Inputs**:

* **1**: ``data`` tensor of arbitrary rank ``r`` and of type *T*. **Required.**

* **2**: ``indices`` tensor with indices of type *T_IND*. The rank of the tensor is equal to the rank of ``data`` tensor. All index values are expected to be within bounds ``[0, s - 1]`` along axis of size ``s``. If multiple indices point to the
  same output location then the order of updating the values is undefined. If an index points to non-existing output
  tensor element or is negative then exception is raised. **Required.**

* **3**: ``updates`` tensor of shape equal to the shape of ``indices`` tensor and of type *T*. **Required.**

* **4**: ``axis`` tensor with scalar or 1D tensor with one element of type *T_AXIS* specifying axis for scatter.
  The value can be in range ``[-r, r - 1]`` where ``r`` is the rank of ``data``. **Required.**

**Outputs**:

* **1**: tensor with shape equal to ``data`` tensor of the type *T*.

**Types**

* *T*: any numeric type.

* *T_IND*: any integer numeric type.

* *T_AXIS*: any integer numeric type.

**Example**

.. code-block:: xml
   :force:

    <layer ... type="ScatterElementsUpdate">
        <input>
            <port id="0">
                <dim>1000</dim>
                <dim>256</dim>
                <dim>7</dim>
                <dim>7</dim>
            </port>
            <port id="1">
                <dim>125</dim>
                <dim>20</dim>
                <dim>7</dim>
                <dim>6</dim>
            </port>
            <port id="2">
                <dim>125</dim>
                <dim>20</dim>
                <dim>7</dim>
                <dim>6</dim>
            </port>
            <port id="3">     <!-- value [0] -->
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="4" precision="FP32">
                <dim>1000</dim>
                <dim>256</dim>
                <dim>7</dim>
                <dim>7</dim>
            </port>
        </output>
    </layer>


