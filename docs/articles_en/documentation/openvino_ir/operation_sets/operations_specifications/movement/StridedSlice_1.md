# StridedSlice {#openvino_docs_ops_movement_StridedSlice_1}

@sphinxdirective

.. meta::
  :description: Learn about StridedSlice-1 - a data movement operation, 
                which can be performed on three required and one optional input tensor.

**Versioned name**: *StridedSlice-1*

**Category**: *Data movement*

**Short description**: *StridedSlice* extracts a strided slice of a tensor.

**Attributes**

* *begin_mask*

  * **Description**: *begin_mask* is a bit mask. *begin_mask[i]* equal to ``1`` means that the corresponding dimension of the ``begin`` input is ignored and the 'real' beginning of the tensor is used along corresponding dimension.
  * **Range of values**: a list of ``0`` s and ``1`` s
  * **Type**: ``int[]``
  * **Default value**: None
  * **Required**: *yes*

* *end_mask*

  * **Description**: *end_mask* is a bit mask. If *end_mask[i]* is ``1``, the corresponding dimension of the ``end`` input is ignored and the real 'end' of the tensor is used along corresponding dimension.
  * **Range of values**: a list of ``0`` s and ``1`` s
  * **Type**: ``int[]``
  * **Default value**: None
  * **Required**: *yes*

* *new_axis_mask*

  * **Description**: *new_axis_mask* is a bit mask. If *new_axis_mask[i]* is ``1``, a length 1 dimension is inserted on the ``i``-th position of input tensor.
  * **Range of values**: a list of ``0`` s and ``1`` s
  * **Type**: ``int[]``
  * **Default value**: ``[0]``
  * **Required**: *no*

* *shrink_axis_mask*

  * **Description**: *shrink_axis_mask* is a bit mask. If *shrink_axis_mask[i]* is ``1``, the dimension on the ``i``-th position is deleted.
  * **Range of values**: a list of ``0`` s and ``1`` s
  * **Type**: ``int[]``
  * **Default value**: ``[0]``
  * **Required**: *no*

* *ellipsis_mask*

  * **Description**: *ellipsis_mask* is a bit mask. It inserts missing dimensions on a position of a non-zero bit.
  * **Range of values**: a list of ``0`` s and ``1``. Only one non-zero bit is allowed.
  * **Type**: ``int[]``
  * **Default value**: ``[0]``
  * **Required**: *no*

**Inputs**:

*   **1**: ``data`` - input tensor to be sliced of type *T* and arbitrary shape. **Required.**

*   **2**: ``begin`` - 1D tensor of type *T_IND* with begin indexes for input tensor slicing. **Required.**
    Out-of-bounds values are silently clamped. If ``begin_mask[i]`` is ``1`` , the value of ``begin[i]`` is ignored and the range of the appropriate dimension starts from ``0``. Negative values mean indexing starts from the end. For example, if ``data=[1,2,3]``, ``begin[0]=-1`` means ``begin[0]=3``.

*   **3**: ``end`` - 1D tensor of type *T_IND* with end indexes for input tensor slicing. **Required.**
    Out-of-bounds values will be silently clamped. If ``end_mask[i]`` is ``1``, the value of ``end[i]`` is ignored and the full range of the appropriate dimension is used instead. Negative values mean indexing starts from the end. For example, if ``data=[1,2,3]``, ``end[0]=-1`` means ``end[0]=3``.

*   **4**: ``stride`` - 1D tensor of type *T_IND* with strides. **Optional.**

**Types**

* *T*: any supported type.
* *T_IND*: any supported integer type.

**Example**
Example of ``begin_mask`` & ``end_mask`` usage.

.. code-block:: xml
   :force:

    <layer ... type="StridedSlice" ...>
        <data begin_mask="0,1,1" ellipsis_mask="0,0,0" end_mask="1,1,0" new_axis_mask="0,0,0" shrink_axis_mask="0,0,0"/>
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>3</dim>
                <dim>4</dim>
            </port>
            <port id="1">
                <dim>2</dim> < !-- begin: [1, 0, 0] -->
            </port>
            <port id="2">
                <dim>2</dim> < !-- end: [0, 0, 2] -->
            </port>
            <port id="3">
                <dim>2</dim> < !-- stride: [1, 1, 1] -->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>1</dim>
                <dim>3</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>


Example of ``new_axis_mask`` usage.

.. code-block:: xml
   :force:

    <layer ... type="StridedSlice" ...>
        <data begin_mask="0,1,1" ellipsis_mask="0,0,0" end_mask="0,1,1" new_axis_mask="1,0,0" shrink_axis_mask="0,0,0"/>
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>3</dim>
                <dim>4</dim>
            </port>
            <port id="1">
                <dim>2</dim>
            </port>
            <port id="2">
                <dim>2</dim>
            </port>
            <port id="3">
                <dim>2</dim>
            </port>
        </input>
        <output>
            <port id="4">
                <dim>1</dim>
                <dim>2</dim>
                <dim>3</dim>
                <dim>4</dim>
            </port>
        </output>
    </layer>

Example of ``shrink_axis_mask`` usage.

.. code-block:: xml
   :force:

    <layer ... type="StridedSlice" ...>
        <data begin_mask="1,0,1,1,1" ellipsis_mask="0,0,0,0,0" end_mask="1,0,1,1,1" new_axis_mask="0,0,0,0,0" shrink_axis_mask="0,1,0,0,0"/>
        <input>
            <port id="0">
                <dim>1</dim>
                <dim>2</dim>
                <dim>384</dim>
                <dim>640</dim>
                <dim>8</dim>
            </port>
            <port id="1">
                <dim>5</dim>
            </port>
            <port id="2">
                <dim>5</dim>
            </port>
            <port id="3">
                <dim>5</dim>
            </port>
        </input>
        <output>
            <port id="4">
                <dim>1</dim>
                <dim>384</dim>
                <dim>640</dim>
                <dim>8</dim>
            </port>
        </output>
    </layer>

@endsphinxdirective

