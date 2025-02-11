ExperimentalDetectronDetectionOutput
====================================


.. meta::
  :description: Learn about ExperimentalDetectronDetectionOutput-6 - an object
                detection operation, which can be performed on four required
                input tensors in OpenVINO.

**Versioned name**: *ExperimentalDetectronDetectionOutput-6*

**Category**: *Object detection*

**Short description**: The *ExperimentalDetectronDetectionOutput* operation performs non-maximum suppression to generate
the detection output using information on location and score predictions.

**Detailed description**: The operation performs the following steps:

1. Applies deltas to boxes sizes [x 1, y 1, x 2, y 2] and takes coordinates of
refined boxes according to the formulas:

``x1_new = ctr_x + (dx - 0.5 * exp(min(d_log_w, max_delta_log_wh))) * box_w``

``y0_new = ctr_y + (dy - 0.5 * exp(min(d_log_h, max_delta_log_wh))) * box_h``

``x1_new = ctr_x + (dx + 0.5 * exp(min(d_log_w, max_delta_log_wh))) * box_w - 1.0``

``y1_new = ctr_y + (dy + 0.5 * exp(min(d_log_h, max_delta_log_wh))) * box_h - 1.0``

* ``box_w`` and ``box_h`` are width and height of box, respectively:

``box_w = x1 - x0 + 1.0``

``box_h = y1 - y0 + 1.0``

* ``ctr_x`` and ``ctr_y`` are center location of a box:

``ctr_x = x0 + 0.5f * box_w``

``ctr_y = y0 + 0.5f * box_h``

* ``dx``, ``dy``, ``d_log_w`` and ``d_log_h`` are deltas calculated according to the formulas below, and ``deltas_tensor`` is a
  second input:

``dx = deltas_tensor[roi_idx, 4 * class_idx + 0] / deltas_weights[0]``

``dy = deltas_tensor[roi_idx, 4 * class_idx + 1] / deltas_weights[1]``

``d_log_w = deltas_tensor[roi_idx, 4 * class_idx + 2] / deltas_weights[2]``

``d_log_h = deltas_tensor[roi_idx, 4 * class_idx + 3] / deltas_weights[3]``

2. If *class_agnostic_box_regression* is ``true`` removes predictions for background classes.
3. Clips boxes to the image.
4. Applies *score_threshold* on detection scores.
5. Applies non-maximum suppression class-wise with *nms_threshold* and returns *post_nms_count* or less detections per class.
6. Returns *max_detections_per_image* detections if total number of detections is more than *max_detections_per_image*; otherwise, returns total number of detections and the output tensor is filled with undefined values for rest output tensor elements.

**Attributes**:

* *score_threshold*

  * **Description**: The *score_threshold* attribute specifies a threshold to consider only detections whose score are larger than the threshold.
  * **Range of values**: non-negative floating-point number
  * **Type**: ``float``
  * **Default value**: None
  * **Required**: *yes*

* *nms_threshold*

  * **Description**: The *nms_threshold* attribute specifies a threshold to be used in the NMS stage.
  * **Range of values**: non-negative floating-point number
  * **Type**: ``float``
  * **Default value**: None
  * **Required**: *yes*

* *num_classes*

  * **Description**: The *num_classes* attribute specifies the number of detected classes.
  * **Range of values**: non-negative integer number
  * **Type**: ``int``
  * **Default value**: None
  * **Required**: *yes*

* *post_nms_count*

  * **Description**: The *post_nms_count* attribute specifies the maximal number of detections per class.
  * **Range of values**: non-negative integer number
  * **Type**: ``int``
  * **Default value**: None
  * **Required**: *yes*

* *max_detections_per_image*

  * **Description**: The *max_detections_per_image* attribute specifies maximal number of detections per image.
  * **Range of values**: non-negative integer number
  * **Type**: ``int``
  * **Default value**: None
  * **Required**: *yes*

* *class_agnostic_box_regression*

  * **Description**: *class_agnostic_box_regression* attribute is a flag that specifies whether to delete background classes or not.
  * **Range of values**:

    * ``true`` means background classes should be deleted
    * ``false`` means background classes should not be deleted
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

* *max_delta_log_wh*

  * **Description**: The *max_delta_log_wh* attribute specifies maximal delta of logarithms for width and height.
  * **Range of values**: floating-point number
  * **Type**: ``float``
  * **Default value**: None
  * **Required**: *yes*

* *deltas_weights*

  * **Description**: The *deltas_weights* attribute specifies weights for bounding boxes sizes deltas.
  * **Range of values**: a list of non-negative floating-point numbers
  * **Type**: ``float[]``
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: A 2D tensor of type *T* with input ROIs, with shape ``[number_of_ROIs, 4]`` providing the ROIs as 4-tuples: [x 1, y 1, x 2, y 2]. The batch dimension of first, second, and third inputs should be the same. **Required.**
* **2**: A 2D tensor of type *T* with shape ``[number_of_ROIs, num_classes * 4]`` providing deltas for input boxes. **Required.**
* **3**: A 2D tensor of type *T* with shape ``[number_of_ROIs, num_classes]`` providing detections scores. **Required.**
* **4**: A 2D tensor of type *T* with shape ``[1, 3]`` contains three elements ``[image_height, image_width, scale_height_and_width]`` providing input image size info. **Required.**

**Outputs**

* **1**: A 2D tensor of type *T* with shape ``[max_detections_per_image, 4]`` providing boxes indices.
* **2**: A 1D tensor of type *T_IND* with shape ``[max_detections_per_image]`` providing classes indices.
* **3**: A 1D tensor of type *T* with shape ``[max_detections_per_image]`` providing scores indices.

**Types**

* *T*: any supported floating-point type.
* *T_IND*: ``int64`` or ``int32``.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="ExperimentalDetectronDetectionOutput" version="opset6">
       <data class_agnostic_box_regression="false" deltas_weights="10.0,10.0,5.0,5.0" max_delta_log_wh="4.135166645050049" max_detections_per_image="100" nms_threshold="0.5" num_classes="81" post_nms_count="2000" score_threshold="0.05000000074505806"/>
       <input>
           <port id="0">
               <dim>1000</dim>
               <dim>4</dim>
           </port>
           <port id="1">
               <dim>1000</dim>
               <dim>324</dim>
           </port>
           <port id="2">
               <dim>1000</dim>
               <dim>81</dim>
           </port>
           <port id="3">
               <dim>1</dim>
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="4" precision="FP32">
               <dim>100</dim>
               <dim>4</dim>
           </port>
           <port id="5" precision="I32">
               <dim>100</dim>
           </port>
           <port id="6" precision="FP32">
               <dim>100</dim>
           </port>
           <port id="7" precision="I32">
               <dim>100</dim>
           </port>
       </output>
   </layer>


