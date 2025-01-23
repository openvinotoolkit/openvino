SegmentMax
==========


.. meta::
  :description: Learn about SegmentMax-16 - an arithmetic operation which computes the maximum values along segments of a tensor.

**Versioned name**: *SegmentMax-16*

**Category**: *Arithmetic*

**Short description**: *SegmentMax-16* operation finds the maximum value in each specified segment of the ``input`` tensor.

**Detailed description**

For each index in ``segment_ids`` the operator gets values from ``data`` input tensor and calculates the maximum value for each segment.

For example ``segments_ids`` with value ``[0,0,0,1,1,3,5,5]`` defines 4 non-empty segments. The other segments are empty. When coupled with a 1D data tensor ``data``, the segments are as follows:

* Segment_0: ``[data[0], data[1], data[2]]``
* Segment_1: ``[data[3], data[4]]``
* Segment_2: ``[]``
* Segment_3: ``[data[5]]``
* Segment_4: ``[]``
* Segment_5: ``[data[6], data[7]]``

When there are no values in a segment, ``output[segment]`` is defined by ``empty_segment_value`` input.

In that case, the output would be ``[max(Segment_0), max(Segment_1), 0, max(Segment_3), 0, max(Segment_5)]``.

**Attributes**:

* **1**: *empty_segment_value*

  * **Description**: The value assigned to segments which are empty. **Required.**
  * **Range of values**: A scalar.
  * **Type**: *T*

**Inputs**

* **1**: ``data`` - ND tensor of type *T*, the numerical data on which SegmentMax operation will be performed. **Required.**

* **2**: ``segment_ids`` - 1D Tensor of sorted non-negative numbers of type *T_IDX*. Its size is equal to the size of the first dimension of the ``data`` input tensor. The values must be smaller than ``num_segments``. **Required.**

* **4**: ``num_segments`` - A scalar value of type *T_IDX* representing the segments count, used for shape inference. **Optional.**

**Outputs**

* **1**: The output tensor has same rank and dimensions as the ``data`` input tensor except for the first dimension which is calculated as ``max(segment_ids) + 1`` 
**Types**

* *T*: any supported numerical data type.
* *T_IDX*: ``int64`` or ``int32``.

**Examples**

*Example 1: 1D input data*

.. code-block:: xml
   :force:

    <layer ... type="SegmentMax" ... >
        <data empty_segment_value="0">
        <input>
            <port id="0" precision="F32">   <!-- data -->
                <dim>8</dim>
            </port>
            <port id="1" precision="I32">   <!-- segment_ids with 4 unique segment IDs -->
                <dim>8</dim> 
            </port>
            <port id="2" precision="I32">   <!-- number of segments -->
                <dim>0</dim> 
            </port>
        </input>
        <output>
            <port id="3" precision="F32">
                <dim>4</dim>
            </port>
        </output>
    </layer>

*Example 2: 2D input data*

.. code-block:: xml
   :force:

    <layer ... type="SegmentMax" ... >
        <data empty_segment_value="0">
        <input>
            <port id="0" precision="I32">   <!-- data -->
                <dim>3</dim>
                <dim>4</dim>
            </port>
            <port id="1" precision="I64">   <!-- segment_ids with 2 unique segment IDs -->
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
