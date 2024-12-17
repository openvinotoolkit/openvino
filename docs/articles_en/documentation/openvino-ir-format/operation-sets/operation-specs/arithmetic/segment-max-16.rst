SegmentMax
===================


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

When there are no values in a segment, ``output[segment]`` is set to 0.

In that case, the output would be ``[max(Segment_0), max(Segment_1), 0, max(Segment_3), 0, max(Segment_5)]``.

**Attributes**:

SegmentMax-16 has no attributes.

**Inputs**

* **1**: *data*

  * **Description**: The numerical data on which SegmentMax operation will be performed. **Required.**
  * **Range of values**: An ND tensor of type *T*.
  * **Type**: *T*

* **2**: *segment_ids*

  * **Description**: controls the data is divided into segments. **Required.**
  * **Range of values**: 1D tensor of non-negative, sorted integer numbers. Its size is equal to the size of the first dimension of the input tensor.
  * **Type**: *T_IDX*

**Outputs**

* **1**: The output tensor of type *T* and the same shape as the ``input`` tensor with the exception for the first dimension, which is equal to the count of unique segment IDs.

**Types**

* *T*: any supported numerical data type.
* *T_IDX*: ``int64`` or ``int32``.

**Examples**

*Example 1: 1D input data*

.. code-block:: xml
   :force:

    <layer ... type="SegmentMax" ... >
        <input>
            <port id="0" precision="F32">   <!-- data -->
                <dim>8</dim>
            </port>
            <port id="1" precision="I32">   <!-- segment_ids with 4 unique segment IDs -->
                <dim>8</dim> 
            </port>
        </input>
        <output>
            <port id="2" precision="F32">
                <dim>4</dim>
            </port>
        </output>
    </layer>

*Example 2: 2D input data*

.. code-block:: xml
   :force:

    <layer ... type="SegmentMax" ... >
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
