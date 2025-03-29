GenerateProposals
=================


.. meta::
  :description: Learn about GenerateProposals-9 - an object detection operation,
                which can be performed on four required input tensors.

**Versioned name**: *GenerateProposals-9*

**Category**: *Object detection*

**Short description**: The *GenerateProposals* operation proposes ROIs and their scores
based on input data for each image in the batch.

**Detailed description**: The operation performs the following steps for each image:

1. Transposes and reshapes predicted bounding boxes deltas and scores to get them into the same dimension order as the
   anchors.
2. Transforms anchors and deltas into proposal bboxes and clips proposal bboxes to an image. The attribute *normalized*
   indicates whether the proposal bboxes are normalized or not.
3. Sorts all ``(proposal, score)`` pairs by score from highest to lowest; order of pairs with equal scores is undefined.
4. Takes top *pre_nms_count* proposals, if total number of proposals is less than *pre_nms_count* takes all proposals.
5. Removes predicted boxes with either height or width < *min_size*.
6. Applies non-maximum suppression with *adaptive_nms_threshold*. The initial value of *adaptive_nms_threshold* is
   *nms_threshold*. If ``nms_eta < 1`` and ``adaptive_threshold > 0.5``, update ``adaptive_threshold *= nms_eta``.
7. Takes and returns top proposals after nms operation. The number of returned proposals in each image is dynamic
   and is specified by output port 3 ``rpnroisnum``. And the max number of proposals in each image is specified
   by attribute *post_nms_count*.

All proposals of the whole batch are concatenated image by image, and distinguishable through outputs.

**Attributes**:

* *min_size*

    * **Description**: The *min_size* attribute specifies minimum box width and height.
    * **Range of values**: non-negative floating-point number
    * **Type**: float
    * **Required**: *yes*

* *nms_threshold*

    * **Description**: The *nms_threshold* attribute specifies threshold to be used in the NMS stage.
    * **Range of values**: non-negative floating-point number
    * **Type**: float
    * **Required**: *yes*

* *pre_nms_count*

    * **Description**: The *pre_nms_count* attribute specifies number of top-n proposals before NMS.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Required**: *yes*

* *post_nms_count*

    * **Description**: The *post_nms_count* attribute specifies number of top-n proposals after NMS.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Required**: *yes*

* *normalized*

    * **Description**: *normalized* is a flag that indicates whether proposal bboxes are normalized or not.
    * **Range of values**: true or false
      * *true* - the bbox coordinates are normalized.
      * *false* - the bbox coordinates are not normalized.
    * **Type**: boolean
    * **Default value**: True
    * **Required**: *no*

* *nms_eta*

    * **Description**: eta parameter for adaptive NMS.
    * **Range of values**: a floating-point number in closed range ``[0, 1.0]``.
    * **Type**: float
    * **Default value**: ``1.0``
    * **Required**: *no*

* *roi_num_type*

    * **Description**: the type of element of output 3 ``rpnroisnum``.
    * **Range of values**: i32, i64
    * **Type**: string
    * **Default value**: ``i64``
    * **Required**: *no*

**Inputs**

* **1**: ``im_info`` - tensor of type *T* and shape ``[num_batches, 3]`` or ``[num_batches, 4]`` providing
  input image info. The image info is layout as ``[image_height, image_width, scale_height_and_width]`` or as
  ``[image_height, image_width, scale_height, scale_width]``. **Required.**
* **2**: ``anchors`` - tensor of type *T* with shape ``[height, width, number_of_anchors, 4]`` providing anchors.
  Each anchor is layouted as ``[xmin, ymin, xmax, ymax]``. **Required.**
* **3**: ``boxesdeltas`` - tensor of type *T* with shape ``[num_batches, number_of_anchors * 4, height, width]``
  providing deltas for anchors. The delta consists of 4 element tuples with layout ``[dx, dy, log(dw), log(dh)]``. **Required.**
* **4**: ``scores`` - tensor of type *T* with shape ``[num_batches, number_of_anchors, height, width]`` providing proposals scores. **Required.**

The ``height`` and ``width`` from inputs ``anchors``, ``boxesdeltas`` and ``scores`` are the height and width of feature maps.

**Outputs**

* **1**: ``rpnrois`` - tensor of type *T* with shape ``[num_rois, 4]`` providing proposed ROIs.
  The proposals are layouted as ``[xmin, ymin, xmax, ymax]``. The ``num_rois`` means the total proposals
  number of all the images in one batch. ``num_rois`` is a dynamic dimension.
* **2**: ``rpnscores`` - tensor of type *T* with shape ``[num_rois]`` providing proposed ROIs scores.
* **3**: ``rpnroisnum`` - tensor of type *roi_num_type* with shape ``[num_batches]`` providing the number
  of proposed ROIs in each image.

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="GenerateProposals" version="opset9">
       <data min_size="0.0" nms_threshold="0.699999988079071" post_nms_count="1000" pre_nms_count="1000" roi_num_type="i32"/>
       <input>
           <port id="0">
               <dim>8</dim>
               <dim>3</dim>
           </port>
           <port id="1">
               <dim>50</dim>
               <dim>84</dim>
               <dim>3</dim>
               <dim>4</dim>
           </port>
           <port id="2">
               <dim>8</dim>
               <dim>12</dim>
               <dim>50</dim>
               <dim>84</dim>
           </port>
           <port id="3">
               <dim>8</dim>
               <dim>3</dim>
               <dim>50</dim>
               <dim>84</dim>
           </port>
       </input>
       <output>
           <port id="4" precision="FP32">
               <dim>-1</dim>
               <dim>4</dim>
           </port>
           <port id="5" precision="FP32">
               <dim>-1</dim>
           </port>
           <port id="6" precision="I32">
               <dim>8</dim>
           </port>
       </output>
   </layer>



