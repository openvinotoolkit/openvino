ExperimentalDetectronGenerateProposalsSingleImage
=================================================


.. meta::
  :description: Learn about ExperimentalDetectronGenerateProposalsSingleImage-6 -
                an object detection operation, which can be performed on four
                required input tensors.

**Versioned name**: *ExperimentalDetectronGenerateProposalsSingleImage-6*

**Category**: *Object detection*

**Short description**: The *ExperimentalDetectronGenerateProposalsSingleImage* operation computes ROIs and their scores based on input data.

**Detailed description**: The operation performs the following steps:

1. Transposes and reshapes predicted bounding boxes deltas and scores to get them into the same order as the anchors.
2. Transforms anchors into proposals using deltas and clips proposals to an image.
3. Removes predicted boxes with either height or width < *min_size*.
4. Sorts all ``(proposal, score)`` pairs by score from highest to lowest; order of pairs with equal scores is undefined.
5. Takes top *pre_nms_count* proposals, if total number of proposals is less than *pre_nms_count* takes all proposals.
6. Applies non-maximum suppression with *nms_threshold*.
7. Takes top *post_nms_count* proposals and returns these top proposals and their scores. If total number of proposals is less than *post_nms_count* returns output tensors filled with zeroes.

**Attributes**:

* *min_size*

  * **Description**: The *min_size* attribute specifies minimum box width and height.
  * **Range of values**: non-negative floating-point number
  * **Type**: ``float``
  * **Default value**: None
  * **Required**: *yes*

* *nms_threshold*

  * **Description**: The *nms_threshold* attribute specifies threshold to be used in the NMS stage.
  * **Range of values**: non-negative floating-point number
  * **Type**: ``float``
  * **Default value**: None
  * **Required**: *yes*

* *pre_nms_count*

  * **Description**: The *pre_nms_count* attribute specifies number of top-n proposals before NMS.
  * **Range of values**: non-negative integer number
  * **Type**: ``int``
  * **Default value**: None
  * **Required**: *yes*

* *post_nms_count*

  * **Description**: The *post_nms_count* attribute specifies number of top-n proposals after NMS.
  * **Range of values**: non-negative integer number
  * **Type**: ``int``
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: A 1D tensor of type *T* with 3 elements ``[image_height, image_width, scale_height_and_width]`` providing input image size info. **Required.**
* **2**: A 2D tensor of type *T* with shape ``[height * width * number_of_channels, 4]`` providing anchors. **Required.**
* **3**: A 3D tensor of type *T* with shape ``[number_of_channels * 4, height, width]`` providing deltas for anchors.
  Height and width for third and fourth inputs should be equal. **Required.**
* **4**: A 3D tensor of type *T* with shape ``[number_of_channels, height, width]`` providing proposals scores. **Required.**

**Outputs**

* **1**: A 2D tensor of type *T* with shape ``[post_nms_count, 4]`` providing ROIs.
* **2**: A 1D tensor of type *T* with shape ``[post_nms_count]`` providing ROIs scores.

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="ExperimentalDetectronGenerateProposalsSingleImage" version="opset6">
       <data min_size="0.0" nms_threshold="0.699999988079071" post_nms_count="1000" pre_nms_count="1000"/>
       <input>
           <port id="0">
               <dim>3</dim>
           </port>
           <port id="1">
               <dim>12600</dim>
               <dim>4</dim>
           </port>
           <port id="2">
               <dim>12</dim>
               <dim>50</dim>
               <dim>84</dim>
           </port>
           <port id="3">
               <dim>3</dim>
               <dim>50</dim>
               <dim>84</dim>
           </port>
       </input>
       <output>
           <port id="4" precision="FP32">
               <dim>1000</dim>
               <dim>4</dim>
           </port>
           <port id="5" precision="FP32">
               <dim>1000</dim>
           </port>
       </output>
   </layer>


