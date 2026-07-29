Proposal
========


.. meta::
  :description: Learn about Proposal-4 - an object detection operation,
                which can be performed on three required input tensors.

**Versioned name**: *Proposal-4*

**Category**: *Object detection*

**Short description**: *Proposal* operation filters bounding boxes and outputs only those with the highest prediction confidence.

**Detailed description**

*Proposal* has three inputs: a 4D tensor of shape ``[num_batches, 2*K, H, W]`` with probabilities whether particular
bounding box corresponds to background or foreground, a 4D tensor of shape ``[num_batches, 4*K, H, W]`` with deltas for each
of the bound box, and a tensor with input image size in the ``[image_height, image_width, scale_height_and_width]`` or
``[image_height, image_width, scale_height, scale_width]`` format. ``K`` is number of anchors and ``H, W`` are height and
width of the feature map. Operation produces two tensors:
the first mandatory tensor of shape ``[batch_size * post_nms_topn, 5]`` with proposed boxes and
the second optional tensor of shape ``[batch_size * post_nms_topn]`` with probabilities (sometimes referred as scores).

*Proposal* layer does the following with the input tensor:

1. Generates initial anchor boxes. Left top corner of all boxes is at (0, 0). Width and height of boxes are calculated from *base_size* with *scale* and *ratio* attributes.
2. For each point in the first input tensor:

   * pins anchor boxes to the image according to the second input tensor that contains four deltas for each box: for *x* and *y* of center, for *width* and for *height*
   * finds out score in the first input tensor

3. Filters out boxes with size less than *min_size*
4. Sorts all proposals (*box*, *score*) by score from highest to lowest
5. Takes top *pre_nms_topn* proposals
6. Calculates intersections for boxes and filter out all boxes with :math:`intersection/union > nms\_thresh`
7. Takes top *post_nms_topn* proposals
8. Returns the results:

   * Top proposals, if there is not enough proposals to fill the whole output tensor, the valid proposals will be terminated with a single -1.
   * Optionally returns probabilities for each proposal, which are not terminated by any special value.

**Attributes**:

* *base_size*

  * **Description**: *base_size* is the size of the anchor to which *scale* and *ratio* attributes are applied.
  * **Range of values**: a positive integer number
  * **Type**: ``int``
  * **Required**: *yes*

* *pre_nms_topn*

  * **Description**: *pre_nms_topn* is the number of bounding boxes before the NMS operation. For example, *pre_nms_topn* equal to 15 means to take top 15 boxes with the highest scores.
  * **Range of values**: a positive integer number
  * **Type**: ``int``
  * **Required**: *yes*

* *post_nms_topn*

  * **Description**: *post_nms_topn* is the number of bounding boxes after the NMS operation. For example, *post_nms_topn* equal to 15 means to take after NMS top 15 boxes with the highest scores.
  * **Range of values**: a positive integer number
  * **Type**: ``int``
  * **Required**: *yes*

* *nms_thresh*

  * **Description**: *nms_thresh* is the minimum value of the proposal to be taken into consideration. For example, *nms_thresh* equal to 0.5 means that all boxes with prediction probability less than 0.5 are filtered out.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

* *feat_stride*

  * **Description**: *feat_stride* is the step size to slide over boxes (in pixels). For example, *feat_stride* equal to 16 means that all boxes are analyzed with the slide 16.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *min_size*

  * **Description**: *min_size* is the minimum size of box to be taken into consideration. For example, *min_size* equal 35 means that all boxes with box size less than 35 are filtered out.
  * **Range of values**: a positive integer number
  * **Type**: ``int``
  * **Required**: *yes*

* *ratio*

  * **Description**: *ratio* is the ratios for anchor generation.
  * **Range of values**: a list of floating-point numbers
  * **Type**: ``float[]``
  * **Required**: *yes*

* *scale*

  * **Description**: *scale* is the scales for anchor generation.
  * **Range of values**: a list of floating-point numbers
  * **Type**: ``float[]``
  * **Required**: *yes*

* *clip_before_nms*

  * **Description**: *clip_before_nms* flag that specifies whether to perform clip bounding boxes before non-maximum suppression or not.
  * **Range of values**: true or false
  * **Type**: ``boolean``
  * **Default value**: true
  * **Required**: *no*

* *clip_after_nms*

  * **Description**: *clip_after_nms* is a flag that specifies whether to perform clip bounding boxes after non-maximum suppression or not.
  * **Range of values**: true or false
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

* *normalize*

  * **Description**: *normalize* is a flag that specifies whether to perform normalization of output boxes to *[0,1]* interval or not.
  * **Range of values**: true or false
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

* *box_size_scale*

  * **Description**: *box_size_scale* specifies the scale factor applied to box sizes before decoding.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: 1.0
  * **Required**: *no*

* *box_coordinate_scale*

  * **Description**: *box_coordinate_scale* specifies the scale factor applied to box coordinates before decoding.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: 1.0
  * **Required**: *no*

* *framework*

  * **Description**: *framework* specifies how the box coordinates are calculated.
  * **Range of values**:

    * "" (empty string) - calculate box coordinates like in Caffe
    * *tensorflow* - calculate box coordinates like in the TensorFlow* Object Detection API models
  * **Type**: string
  * **Default value**: "" (empty string)
  * **Required**: *no*

**Inputs**:

*   **1**: 4D tensor of type *T* and shape ``[batch_size, 2*K, H, W]`` with class prediction scores. **Required.**

*   **2**: 4D tensor of type *T* and shape ``[batch_size, 4*K, H, W]`` with deltas for each bounding box. **Required.**

*   **3**: 1D tensor of type *T* with 3 or 4 elements:  ``[image_height, image_width, scale_height_and_width]`` or ``[image_height, image_width, scale_height, scale_width]``. **Required.**

**Outputs**

*   **1**: tensor of type *T* and shape ``[batch_size * post_nms_topn, 5]``.

*   **2**: tensor of type *T* and shape ``[batch_size * post_nms_topn]`` with probabilities.

**Types**

* *T*: floating-point type.

**Example**


.. code-block:: xml
   :force:

   <layer ... type="Proposal" ... >
       <data base_size="16" feat_stride="8" min_size="16" nms_thresh="1.0" normalize="0" post_nms_topn="1000" pre_nms_topn="1000" ratio="1" scale="1,2"/>
       <input>
           <port id="0">
               <dim>7</dim>
               <dim>4</dim>
               <dim>28</dim>
               <dim>28</dim>
           </port>
           <port id="1">
               <dim>7</dim>
               <dim>8</dim>
               <dim>28</dim>
               <dim>28</dim>
           </port>
           <port id="2">
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="3" precision="FP32">
               <dim>7000</dim>
               <dim>5</dim>
           </port>
           <port id="4" precision="FP32">
               <dim>7000</dim>
           </port>
       </output>
   </layer>




