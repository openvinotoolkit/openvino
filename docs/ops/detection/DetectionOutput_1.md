## DetectionOutput <a name="DetectionOutput"></a> {#openvino_docs_ops_detection_DetectionOutput_1}

**Versioned name**: *DetectionOutput-1*

**Category**: *Object detection*

**Short description**: *DetectionOutput* performs non-maximum suppression to generate the detection output using information on location and confidence predictions.

**Detailed description**: [Reference](https://arxiv.org/pdf/1512.02325.pdf). The layer has 3 mandatory inputs: tensor with box logits, tensor with confidence predictions and tensor with box coordinates (proposals). It can have 2 additional inputs with additional confidence predictions and box coordinates described in the [article](https://arxiv.org/pdf/1711.06897.pdf). The 5-input version of the layer is supported with Myriad plugin only. The output tensor contains information about filtered detections described with 7 element tuples: *[batch_id, class_id, confidence, x_1, y_1, x_2, y_2]*. The first tuple with *batch_id* equal to *-1* means end of output.

At each feature map cell, *DetectionOutput* predicts the offsets relative to the default box shapes in the cell, as well as the per-class scores that indicate the presence of a class instance in each of those boxes. Specifically, for each box out of k at a given location, *DetectionOutput* computes class scores and the four offsets relative to the original default box shape. This results in a total of \f$(c + 4)k\f$ filters that are applied around each location in the feature map, yielding \f$(c + 4)kmn\f$ outputs for a *m \* n* feature map.

**Attributes**:

* *num_classes*

  * **Description**: number of classes to be predicted
  * **Range of values**: positive integer number
  * **Type**: int
  * **Default value**: None
  * **Required**: *yes*

* *background_label_id*

  * **Description**: background label id. If there is no background class, set it to -1.
  * **Range of values**: integer values
  * **Type**: int
  * **Default value**: 0
  * **Required**: *no*

* *top_k*

  * **Description**: maximum number of results to be kept per batch after NMS step. -1 means keeping all bounding boxes.
  * **Range of values**: integer values
  * **Type**: int
  * **Default value**: -1
  * **Required**: *no*

* *variance_encoded_in_target*

  * **Description**: *variance_encoded_in_target* is a flag that denotes if variance is encoded in target. If flag is false then it is necessary to adjust the predicted offset accordingly.
  * **Range of values**: false or true
  * **Type**: boolean
  * **Default value**: false
  * **Required**: *no*

* *keep_top_k*

  * **Description**: maximum number of bounding boxes per batch to be kept after NMS step. -1 means keeping all bounding boxes after NMS step.
  * **Range of values**: integer values
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *code_type*

  * **Description**: type of coding method for bounding boxes
  * **Range of values**: "caffe.PriorBoxParameter.CENTER_SIZE", "caffe.PriorBoxParameter.CORNER"
  * **Type**: string
  * **Default value**: "caffe.PriorBoxParameter.CORNER"
  * **Required**: *no*

* *share_location*

  * **Description**: *share_location* is a flag that denotes if bounding boxes are shared among different classes.
  * **Range of values**: 0 or 1
  * **Type**: int
  * **Default value**: 1
  * **Required**: *no*

* *nms_threshold*

  * **Description**: threshold to be used in the NMS stage
  * **Range of values**: floating point values
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

* *confidence_threshold*

  * **Description**: only consider detections whose confidences are larger than a threshold. If not provided, consider all boxes.
  * **Range of values**: floating point values
  * **Type**: float
  * **Default value**: 0
  * **Required**: *no*

* *clip_after_nms*

  * **Description**: *clip_after_nms* flag that denotes whether to perform clip bounding boxes after non-maximum suppression or not.
  * **Range of values**: 0 or 1
  * **Type**: int
  * **Default value**: 0
  * **Required**: *no*

* *clip_before_nms*

  * **Description**: *clip_before_nms* flag that denotes whether to perform clip bounding boxes before non-maximum suppression or not.
  * **Range of values**: 0 or 1
  * **Type**: int
  * **Default value**: 0
  * **Required**: *no*

* *decrease_label_id*

  * **Description**: *decrease_label_id* flag that denotes how to perform NMS.
  * **Range of values**:
    * 0 - perform NMS like in Caffe\*.
    * 1 - perform NMS like in MxNet\*.
  * **Type**: int
  * **Default value**: 0
  * **Required**: *no*

* *normalized*

  * **Description**: *normalized* flag that denotes whether input tensors with boxes are normalized. If tensors are not normalized then *input_height* and *input_width* attributes are used to normalize box coordinates.
  * **Range of values**: 0 or 1
  * **Type**: int
  * **Default value**: 0
  * **Required**: *no*

* *input_height (input_width)*

  * **Description**: input image height (width). If the *normalized* is 1 then these attributes are not used.
  * **Range of values**: positive integer number
  * **Type**: int
  * **Default value**: 1
  * **Required**: *no*

* *objectness_score*

  * **Description**: threshold to sort out confidence predictions. Used only when the *DetectionOutput* layer has 5 inputs.
  * **Range of values**: non-negative float number
  * **Type**: float
  * **Default value**: 0
  * **Required**: *no*
  
**Inputs**

* **1**: 2D input tensor with box logits. Required.
* **2**: 2D input tensor with class predictions. Required.
* **3**: 3D input tensor with proposals. Required.
* **4**: 2D input tensor with additional class predictions information described in the [article](https://arxiv.org/pdf/1711.06897.pdf). Optional.
* **5**: 2D input tensor with additional box predictions information described in the [article](https://arxiv.org/pdf/1711.06897.pdf). Optional.

**Example**

```xml
<layer ... type="DetectionOutput" ... >
    <data num_classes="21" share_location="1" background_label_id="0" nms_threshold="0.450000" top_k="400" input_height="1" input_width="1" code_type="caffe.PriorBoxParameter.CENTER_SIZE" variance_encoded_in_target="0" keep_top_k="200" confidence_threshold="0.010000"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```