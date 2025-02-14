SegmentMax
==========


.. meta::
  :description: Learn about SegmentMax-16 - an arithmetic operation which computes the maximum values along segments of a tensor.

**Versioned name**: *SegmentMax-16*

**Category**: *Arithmetic*

**Short description**: *SegmentMax-16* operation finds the maximum value in each specified segment of the ``input`` tensor.

**Detailed description**

For each index in ``segment_ids`` the operator gets values from ``data`` input tensor and calculates the maximum value for each segment.

For example ``segments_ids`` with value ``[0,0,0,1,1,3,5,5]`` defines 4 non-empty segments. ``num_segments`` is not given. When coupled with a 1D data tensor ``data``, the segments are as follows:

* Segment_0: ``[data[0], data[1], data[2]]``
* Segment_1: ``[data[3], data[4]]``
* Segment_2: ``[]``
* Segment_3: ``[data[5]]``
* Segment_4: ``[]``
* Segment_5: ``[data[6], data[7]]``

When there are no values in a segment, ``output[segment]`` is defined by ``fill_mode`` attribute.

For ``fill_mode`` equal to ``ZERO`` , the operation output would be ``[max(Segment_0), max(Segment_1), 0, max(Segment_3), 0, max(Segment_5)]``.

**Attributes**:

* **1**: *fill_mode*

  * **Description**: Responsible for the value assigned to segments which are empty. **Required.**
  * **Range of values**: Name of the mode in string format:

    * ``ZERO`` - the empty segments are filled with zeros.
    * ``LOWEST`` - the empty segments are filled with the lowest value of the data type *T*.
  * **Type**: ``string``

**Inputs**

* **1**: ``data`` - ND tensor of type *T*, the numerical data on which SegmentMax operation will be performed. **Required.**

* **2**: ``segment_ids`` - 1D Tensor of sorted non-negative numbers of type *T_IDX1*. Its size is equal to the size of the first dimension of the ``data`` input tensor. **Required.**

* **3**: ``num_segments`` - A scalar value of type *T_IDX2* representing the segments count. If ``num_segments < max(segment_ids) + 1`` then the extra segments defined in ``segment_ids`` are not included in the output. If If ``num_segments > max(segment_ids) + 1`` then the output is padded with empty segments. Defaults to ``max(segment_ids) + 1``. **Optional.**

**Outputs**

* **1**: The output tensor has same rank and dimensions as the ``data`` input tensor except for the first dimension which is equal to the value of ``num_segments``.

**Types**

* *T*: any supported numerical data type.
* *T_IDX1*, *T_IDX2*: ``int64`` or ``int32``.

**Examples**

*Example 1: num_segments < max(segment_ids) + 1*

.. code-block:: xml
   :force:

    <layer ... type="SegmentMax" ... >
        <data empty_segment_value="ZERO">
        <input>
            <port id="0" precision="F32">   <!-- data -->
                <dim>5</dim>
            </port>
            <port id="1" precision="I32">   <!-- segment_ids with 4 segments: [0, 0, 2, 3, 3] -->
                <dim>5</dim> 
            </port>
            <port id="2" precision="I64">   <!-- number of segments: 2 -->
                <dim>0</dim> 
            </port>
        </input>
        <output>
            <port id="3" precision="F32">
                <dim>2</dim>
            </port>
        </output>
    </layer>

*Example 2: num_segments > max(segment_ids) + 1*

.. code-block:: xml
   :force:

    <layer ... type="SegmentMax" ... >
        <data empty_segment_value="ZERO">
        <input>
            <port id="0" precision="F32">   <!-- data -->
                <dim>5</dim>
            </port>
            <port id="1" precision="I32">   <!-- segment_ids with 4 segments: [0, 0, 2, 3, 3] -->
                <dim>5</dim> 
            </port>
            <port id="2" precision="I64">   <!-- number of segments: 8 -->
                <dim>0</dim> 
            </port>
        </input>
        <output>
            <port id="3" precision="F32">
                <dim>8</dim>
            </port>
        </output>
    </layer>

*Example 3: 2D input data, no num_segments*

.. code-block:: xml
   :force:

    <layer ... type="SegmentMax" ... >
        <data empty_segment_value="LOWEST">
        <input>
            <port id="0" precision="I32">   <!-- data -->
                <dim>3</dim>
                <dim>4</dim>
            </port>
            <port id="1" precision="I64">   <!-- segment_ids with 2 segments: [0, 1, 1] -->
                <dim>3</dim>
            </port>
        </input>
        <output>
            <port id="2" precision="I32">
                <dim>2</dim>
                <dim>4</dim>
            </port>
        </output>
    </layer>
