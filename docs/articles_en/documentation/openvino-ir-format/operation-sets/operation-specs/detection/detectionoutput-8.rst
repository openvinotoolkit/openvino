DetectionOutput
===============


.. meta::
  :description: Learn about DetectionOutput-8 - an object detection operation, which
                can be performed on three mandatory and two additional tensors in OpenVINO.

**Versioned name**: *DetectionOutput-8*

**Category**: *Object detection*

**Short description**: *DetectionOutput* performs non-maximum suppression to generate the detection output using information on location and
confidence predictions.

**Detailed description**: `Reference <https://arxiv.org/pdf/1512.02325.pdf>`__ . The layer has 3 mandatory inputs: tensor with box logits, tensor with confidence predictions and tensor with box coordinates (proposals). It can have 2 additional inputs with additional confidence predictions and box coordinates described in the `article <https://arxiv.org/pdf/1711.06897.pdf>`__ . The output tensor contains information about filtered detections described with 7 element tuples: ``[batch_id, class_id, confidence, x_1, y_1, x_2, y_2]``. The first tuple with ``batch_id`` equal to ``-1`` means end of output.

At each feature map cell, *DetectionOutput* predicts the offsets relative to the default box shapes in the cell, as well as the per-class scores that indicate the presence of a class instance in each of those boxes. Specifically, for each box out of k at a given location, *DetectionOutput* computes class scores and the four offsets relative to the original default box shape. This results in a total of :math:`(c + 4)k` filters that are applied around each location in the feature map, yielding :math:`(c + 4)kmn` outputs for a *m \* n* feature map.

**Attributes**:

.. note::

   *num_classes*, a number of classes attribute, presents in :doc:`DetectionOutput_1 <detectionoutput-1>` has been removed. It can be computed as ``cls_pred_shape[-1] // num_prior_boxes`` where ``cls_pred_shape`` and ``num_prior_boxes`` are class predictions tensor shape and a number of prior boxes.

* *background_label_id*

  * **Description**: background label id. If there is no background class, set it to -1.
  * **Range of values**: integer values
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* *top_k*

  * **Description**: maximum number of results to be kept per batch after NMS step. -1 means keeping all bounding boxes.
  * **Range of values**: integer values
  * **Type**: ``int``
  * **Default value**: -1
  * **Required**: *no*

* *variance_encoded_in_target*

  * **Description**: *variance_encoded_in_target* is a flag that denotes if variance is encoded in target. If flag is false then it is necessary to adjust the predicted offset accordingly.
  * **Range of values**: false or true
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

* *keep_top_k*

  * **Description**: maximum number of bounding boxes per batch to be kept after NMS step. -1 means keeping all bounding boxes after NMS step.
  * **Range of values**: integer values
  * **Type**: ``int[]``
  * **Required**: *yes*

* *code_type*

  * **Description**: type of coding method for bounding boxes
  * **Range of values**: "caffe.PriorBoxParameter.CENTER_SIZE", "caffe.PriorBoxParameter.CORNER"
  * **Type**: ``string``
  * **Default value**: "caffe.PriorBoxParameter.CORNER"
  * **Required**: *no*

* *share_location*

  * **Description**: *share_location* is a flag that denotes if bounding boxes are shared among different classes.
  * **Range of values**: false or true
  * **Type**: ``boolean``
  * **Default value**: true
  * **Required**: *no*

* *nms_threshold*

  * **Description**: threshold to be used in the NMS stage
  * **Range of values**: floating-point values
  * **Type**: ``float``
  * **Required**: *yes*

* *confidence_threshold*

  * **Description**: only consider detections whose confidences are larger than a threshold. If not provided, consider all boxes.
  * **Range of values**: floating-point values
  * **Type**: ``float``
  * **Default value**: 0
  * **Required**: *no*

* *clip_after_nms*

  * **Description**: *clip_after_nms* flag that denotes whether to perform clip bounding boxes after non-maximum suppression or not.
  * **Range of values**: false or true
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

* *clip_before_nms*

  * **Description**: *clip_before_nms* flag that denotes whether to perform clip bounding boxes before non-maximum suppression or not.
  * **Range of values**: false or true
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

* *decrease_label_id*

  * **Description**: *decrease_label_id* flag that denotes how to perform NMS.
  * **Range of values**:

    * false - perform NMS like in Caffe.
    * true - perform NMS like in Apache MxNet.
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

* *normalized*

  * **Description**: *normalized* flag that denotes whether input tensor with proposal boxes is normalized. If tensor is not normalized then *input_height* and *input_width* attributes are used to normalize box coordinates.
  * **Range of values**: false or true
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

* *input_height (input_width)*

  * **Description**: input image height (width). If the *normalized* is 1 then these attributes are not used.
  * **Range of values**: positive integer number
  * **Type**: ``int``
  * **Default value**: 1
  * **Required**: *no*

* *objectness_score*

  * **Description**: threshold to sort out confidence predictions. Used only when the *DetectionOutput* layer has 5 inputs.
  * **Range of values**: non-negative float number
  * **Type**: ``float``
  * **Default value**: 0
  * **Required**: *no*

**Inputs**

* **1**: 2D input tensor with box logits with shape ``[N, num_prior_boxes * num_loc_classes * 4]`` and type *T*. ``num_loc_classes`` is equal to ``num_classes`` when ``share_location`` is 0 or it's equal to 1 otherwise. **Required.**
* **2**: 2D input tensor with class predictions with shape ``[N, num_prior_boxes * num_classes]`` and type *T*. **Required.**
* **3**: 3D input tensor with proposals with shape ``[priors_batch_size, 1, num_prior_boxes * prior_box_size]`` or ``[priors_batch_size, 2, num_prior_boxes * prior_box_size]``. ``priors_batch_size`` is either 1 or ``N``. Size of the second dimension depends on ``variance_encoded_in_target``. If ``variance_encoded_in_target`` is equal to 0, the second dimension equals to 2 and variance values are provided for each boxes coordinates. If ``variance_encoded_in_target`` is equal to 1, the second dimension equals to 1 and this tensor contains proposals boxes only. ``prior_box_size`` is equal to 4 when ``normalized`` is set to 1 or it's equal to 5 otherwise. **Required.**
* **4**: 2D input tensor with additional class predictions information described in the `article <https://arxiv.org/pdf/1711.06897.pdf>`__. Its shape must be equal to ``[N, num_prior_boxes * 2]``. **Optional.**
* **5**: 2D input tensor with additional box predictions information described in the `article <https://arxiv.org/pdf/1711.06897.pdf>`__. Its shape must be equal to first input tensor shape. **Optional.**

**Outputs**

* **1**: 4D output tensor with type *T*. Its shape depends on ``keep_top_k`` or ``top_k`` being set. It ``keep_top_k[0]`` is greater than zero, then the shape is ``[1, 1, N * keep_top_k[0], 7]``. If ``keep_top_k[0]`` is set to -1 and ``top_k`` is greater than zero, then the shape is ``[1, 1, N * top_k * num_classes, 7]``. Otherwise, the output shape is equal to ``[1, 1, N * num_classes * num_prior_boxes, 7]``.

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="DetectionOutput" version="opset8">
       <data background_label_id="1" code_type="caffe.PriorBoxParameter.CENTER_SIZE" confidence_threshold="0.019999999552965164" input_height="1" input_width="1" keep_top_k="200" nms_threshold="0.44999998807907104" normalized="true" share_location="true" top_k="200" variance_encoded_in_target="false" clip_after_nms="false" clip_before_nms="false" objectness_score="0" decrease_label_id="false"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>5376</dim>
           </port>
           <port id="1">
               <dim>1</dim>
               <dim>2688</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>2</dim>
               <dim>5376</dim>
           </port>
       </input>
       <output>
           <port id="3" precision="FP32">
               <dim>1</dim>
               <dim>1</dim>
               <dim>200</dim>
               <dim>7</dim>
           </port>
       </output>
   </layer>


