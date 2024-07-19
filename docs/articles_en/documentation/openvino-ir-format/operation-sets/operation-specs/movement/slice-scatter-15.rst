.. {#openvino_docs_ops_movement_SliceScatter_15}

SliceScatter
===============


.. meta::
  :description: Learn about SliceScatter-15 - a data movement operation, which can be 
                performed on six required input tensors.

**Versioned name**: *SliceScatter-15*

**Category**: *Data movement*

**Short description**: Creates copy of ``data`` tensor and applies elements from ``updates`` using slicing parametrized by ``start``, ``stop`` and ``step`` over axis ``axis``.

**Detailed description**:

Creates copy of ``data`` tensor and applies elements from ``updates`` using slicing parametrized by ``start``, ``stop`` and ``step`` over ``data`` axis ``axis``.

General logic of updating data tensor for 1D, 2D and 3D tensors:

.. code-block:: py

  indices_range = range(start, stop, step)
  # For 1D data tensor:
  output[indices_range[i]] = updates[i], axis = 0
  # For 2D data tensor:
  output[indices_range[i]][j] = updates[i][j] if axis = 0
  output[i][indices_range[j]] = updates[i][j] if axis = 1
  # For 3D data tensor:
  output[indices_range[i]][j][k] = updates[i][j][k] if axis = 0
  output[i][indices_range[j]][k] = updates[i][j][k] if axis = 1
  output[i][j][indices_range[k]] = updates[i][j][k] if axis = 2


**Attributes**: *SliceScatter* does not have attributes.

**Inputs**:

* **1**: ``data`` tensor of arbitrary rank ``r`` >= 1 and of type *T*. **Required.**

* **2**: ``updates`` - tensor of type *T* and same shape as ``data`` except axis set by ``axes`` where it should be the length of update slice. **Required.**

* **3**: ``start`` - 0D or single element 1D tensor of type *T_IND*.

  Defines the starting coordinate of the update slice in the ``data`` tensor at ``axis`` dimension.
  A negative index value represents counting elements from the end of that dimension.
  A value larger than the size of a dimension is silently clamped. **Required.**

* **4**: ``stop`` - 0D or single element 1D tensor of type *T_IND*.

  Defines the coordinate of the opposite vertex of the update slice, or where the update slice ends.
  Stop indexes are exclusive, which means values lying on the ending edge are
  not included in the range of indices.
  To create slice to the end of a dimension of unknown size ``INT_MAX``
  may be used (or ``INT_MIN`` if slicing backwards). **Required.**

* **5**: ``step`` - 0D or single element 1D tensor of type *T_IND*.

  Integer value that specifies the increment between each index used in slicing.
  Value cannot be ``0``, negative value indicates slicing backwards. **Required.**

* **6**: ``axes`` - 0D or single element 1D tensor of type *T_AXIS*.

  Integer value that specifies which dimensions the values in ``start`` and ``stop`` apply to.
  Negative value means counting dimensions from the end. The range is ``[-r, r - 1]``, where ``r`` is the rank of the ``data`` input tensor. **Required.**

**Outputs**:

*   **1**: tensor with shape equal to ``data`` tensor of the type *T*.

**Types**

* *T*: any numeric type.
* *T_IND*: any supported integer type.
* *T_AXIS*: any supported integer type.

**Example**

*Example 1: Fill slice over axis==0.*

.. code-block:: xml

    <layer ... type="SliceScatter">
        <input>
            <port id="0" precision="FP32">  <!-- data -->
                <dim>2</dim>
                <dim>5</dim>  <!-- values: [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]] -->
            </port>
            <port id="1" precision="FP32">  <!-- updates -->
                <dim>1</dim>
                <dim>5</dim>  <!-- values: [[10, 20, 30, 40, 50]] -->
            </port>
            <port id="2" precision="I32">  <!-- start -->
                <dim>1</dim>  <!-- values: [0] -->
            </port>
            <port id="3" precision="I32">  <!-- stop -->
                <dim>1</dim>  <!-- values: [1] -->
            </port>
            <port id="4" precision="I32">  <!-- step -->
                <dim>1</dim>  <!-- values: [1] -->
            </port>
            <port id="5" precision="I32">  <!-- axis -->
                <dim>1</dim>  <!-- values: [0] -->
            </port>
        </input>
        <output>
            <port id="6" precision="FP32">
                <dim>2</dim>
                <dim>5</dim>  <!-- values: [[10, 20, 30, 40, 50], [5, 6, 7, 8, 9]] -->
            </port>
        </output>
    </layer>

*Example 2: Update every second value over axis==1, clamp values of start and stop.*

.. code-block:: xml

    <layer ... type="SliceScatter">
        <input>
            <port id="0" precision="FP32">  <!-- data -->
                <dim>2</dim>
                <dim>5</dim>  <!-- values: [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]] -->
            </port>
            <port id="1" precision="FP32">  <!-- updates -->
                <dim>2</dim>
                <dim>3</dim>  <!-- values: [[10, 20, 30], [40, 50, 60]] -->
            </port>
            <port id="2" precision="I32">  <!-- start -->
                <dim>1</dim>  <!-- values: [-25], silently clamped to 0 -->
            </port>
            <port id="3" precision="I32">  <!-- stop -->
                <dim>1</dim>  <!-- values: [25], silently clamped to 5 -->
            </port>
            <port id="4" precision="I32">  <!-- step -->
                <dim>1</dim>  <!-- values: [2] -->
            </port>
            <port id="5" precision="I32">  <!-- axis -->
                <dim>1</dim>  <!-- values: [1] -->
            </port>
        </input>
        <output>
            <port id="6" precision="FP32">
                <dim>2</dim>
                <dim>5</dim>  <!-- values: [[10, 1, 20, 3, 30], [40, 6, 50, 8, 60]] -->
            </port>
        </output>
    </layer>
