RegionYolo
==========


.. meta::
  :description: Learn about RegionYolo-1 - an object detection operation,
                which can be performed on a 4D input tensor.

**Versioned name**: *RegionYolo-1*

**Category**: *Object detection*

**Short description**: *RegionYolo* computes the coordinates of regions with probability for each class.

**Detailed description**: This operation is directly mapped to the `YOLO9000: Better, Faster, Stronger <https://arxiv.org/pdf/1612.08242.pdf>`__ paper.

**Attributes**:

* *anchors*

  * **Description**: *anchors* codes a flattened list of pairs ``[width, height]`` that codes prior box sizes. This attribute is not used in output computation, but it is required for post-processing to restore real box coordinates.
  * **Range of values**: list of any length of positive floating-point number
  * **Type**: ``float[]``
  * **Default value**: None
  * **Required**: *no*

* *axis*

  * **Description**: starting axis index in the input tensor ``data`` shape that will be flattened in the output; the end of flattened range is defined by ``end_axis`` attribute.
  * **Range of values**: ``-rank(data) .. rank(data)-1``
  * **Type**: ``int``
  * **Required**: *yes*

* *coords*

  * **Description**: *coords* is the number of coordinates for each region.
  * **Range of values**: an integer
  * **Type**: ``int``
  * **Required**: *yes*

* *classes*

  * **Description**: *classes* is the number of classes for each region.
  * **Range of values**: an integer
  * **Type**: ``int``
  * **Required**: *yes*

* *end_axis*

  * **Description**: ending axis index in the input tensor ``data`` shape that will be flattened in the output; the beginning of the flattened range is defined by ``axis`` attribute.
  * **Range of values**: ``-rank(data)..rank(data)-1``
  * **Type**: ``int``
  * **Required**: *yes*

* *num*

  * **Description**: *num* is the number of regions.
  * **Range of values**: an integer
  * **Type**: ``int``
  * **Required**: *yes*

* *do_softmax*

  * **Description**: *do_softmax* is a flag that specifies the inference method and affects how the number of regions is determined. It also affects output shape. If it is 0, then output shape is 4D, and 2D otherwise.
  * **Range of values**:

    * *false* - do not perform softmax
    * *true* - perform softmax
  * **Type**: ``boolean``
  * **Default value**: true
  * **Required**: *no*

* *mask*

  * **Description**: *mask* specifies the number of regions. Use this attribute instead of *num* when *do_softmax* is equal to 0.
  * **Range of values**: a list of integers
  * **Type**: ``int[]``
  * **Default value**: ``[]``
  * **Required**: *no*

**Inputs**:

*   **1**: ``data`` - 4D tensor of type *T* and shape ``[N, C, H, W]``. **Required.**

**Outputs**:

*   **1**: tensor of type *T* and rank 4 or less that codes detected regions. Refer to the `YOLO9000: Better, Faster, Stronger <https://arxiv.org/pdf/1612.08242.pdf>`__ paper to decode the output as boxes. ``anchors`` should be used to decode real box coordinates. If ``do_softmax`` is set to ``0``, then the output shape is ``[N, (classes + coords + 1) * len(mask), H, W]``. If ``do_softmax`` is set to ``1``, then output shape is partially flattened and defined in the following way:

``flat_dim = data.shape[axis] * data.shape[axis+1] * ... * data.shape[end_axis]``
``output.shape = [data.shape[0], ..., data.shape[axis-1], flat_dim, data.shape[end_axis + 1], ...]``

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <!-- YOLO V3 example -->
   <layer type="RegionYolo" ... >
       <data anchors="10,14,23,27,37,58,81,82,135,169,344,319" axis="1" classes="80" coords="4" do_softmax="0" end_axis="3" mask="0,1,2" num="6"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>255</dim>
               <dim>26</dim>
               <dim>26</dim>
           </port>
       </input>
       <output>
           <port id="0">
               <dim>1</dim>
               <dim>255</dim>
               <dim>26</dim>
               <dim>26</dim>
           </port>
       </output>
   </layer>

   <!-- YOLO V2 Example -->
   <layer type="RegionYolo" ... >
       <data anchors="1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52" axis="1" classes="20" coords="4" do_softmax="1" end_axis="3" num="5"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>125</dim>
               <dim>13</dim>
               <dim>13</dim>
           </port>
       </input>
       <output>
           <port id="0">
               <dim>1</dim>
               <dim>21125</dim>
           </port>
       </output>
   </layer>


