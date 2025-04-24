SliceScatter
===============


.. meta::
  :description: Learn about SliceScatter-15 - a data movement operation, which can be
                performed on five required input tensors.

**Versioned name**: *SliceScatter-15*

**Category**: *Data movement*

**Short description**: Creates copy of ``data`` tensor and applies elements from ``updates`` using slicing parametrized by ``start``, ``stop`` and ``step`` over ``axes`` dimensions.

**Detailed description**:

The operation produces a copy of ``data`` tensor and updates it using values specified by ``updates`` at specific slices parametrized by ``start``, ``stop`` and ``step`` values over corresponding ``axes`` dimensions.
The output shape and type is the same as ``data``. Number of elements for ``start``, ``stop``, ``step`` and ``axes`` is required to be equal. Values in ``axes`` are required to be unique.

Operator SliceScatter-15 is an equivalent to following NumPy snippet:

.. code-block:: py

    def slice_scatter_15(
        data: np.ndarray,
        updates: np.ndarray,
        start: List[int],
        stop: List[int],
        step: List[int],
        axes: Optional[List[int]] = None,
    ):
        out = np.copy(data)
        if axes is None:
            axes = list(range(len(start)))
        slice_list = [slice(None)] * data.ndim
        for slice_start, slice_stop, slice_step, slice_axis in zip(start, stop, step, axes):
            slice_list[slice_axis] = slice(slice_start, slice_stop, slice_step)
        out[tuple(slice_list)] = updates
        return out

**Attributes**: *SliceScatter* does not have attributes.

**Inputs**:

* **1**: ``data`` tensor of arbitrary rank ``r`` >= 1 and of type *T*. **Required.**

* **2**: ``updates`` - tensor of type *T* and same shape as ``data`` except axes set by ``axes`` where dimensions should be equal to the length of corresponding update slices. **Required.**

* **3**: ``start`` - 1D tensor of type *T_IND*.

  Defines the starting coordinate of the update slice in the ``data`` tensor at ``axes`` dimension.
  A negative index value represents counting elements from the end of that dimension.
  A value larger than the size of a dimension is silently clamped. **Required.**

* **4**: ``stop`` - 1D tensor of type *T_IND*.

  Defines the coordinate of the opposite vertex of the update slice, or where the update slice ends.
  Stop indexes are exclusive, which means values lying on the ending edge are not updated.
  A value larger than the size of a dimension is silently clamped.
  To create slice to the end of a dimension of unknown size ``INT_MAX``
  may be used (or ``INT_MIN`` if slicing backwards). **Required.**

* **5**: ``step`` - 1D tensor of type *T_IND*.

  Integer value that specifies the increment between each index used in slicing.
  Value cannot be ``0``, negative value indicates slicing backwards. **Required.**

* **6**: ``axes`` - 1D tensor of type *T_AXIS*.

  Optional 1D tensor indicating which dimensions the values in ``start``, ``stop`` and ``step`` apply to.
  Negative value means counting dimensions from the end. The range is ``[-r, r - 1]``, where ``r`` is the rank of the ``data`` input tensor.
  Values are required to be unique.
  Default value: ``[0, 1, 2, ..., start.shape[0] - 1]``. **Optional.**

Number of elements in ``start``, ``stop``, ``step``, and ``axes`` inputs are required to be equal.

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
            <port id="5" precision="I32">  <!-- axes -->
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
            <port id="5" precision="I32">  <!-- axes -->
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

*Example 3: Update every second value over both axes with different slice starts.*

.. code-block:: xml

    <layer ... type="SliceScatter">
        <input>
            <port id="0" precision="FP32">  <!-- data -->
                <dim>3</dim>
                <dim>5</dim>  <!-- values: [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]] -->
            </port>
            <port id="1" precision="FP32">  <!-- updates -->
                <dim>2</dim>
                <dim>2</dim>  <!-- values: [[50, 60], [70, 80]] -->
            </port>
            <port id="2" precision="I32">  <!-- start -->
                <dim>1</dim>  <!-- values: [0, 1] -->
            </port>
            <port id="3" precision="I32">  <!-- stop -->
                <dim>1</dim>  <!-- values: [3, 5] -->
            </port>
            <port id="4" precision="I32">  <!-- step -->
                <dim>1</dim>  <!-- values: [2, 2] -->
            </port>
        </input>
        <output>
            <port id="5" precision="FP32">
                <dim>3</dim>
                <dim>5</dim>  <!-- values: [[0, 50, 2, 60, 4], [5, 6, 7, 8, 9], [10, 70, 12, 80, 14]] -->
            </port>
        </output>
    </layer>
