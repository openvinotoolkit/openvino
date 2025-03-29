NonMaxSuppression
=================


.. meta::
  :description: Learn about NonMaxSuppression-4 - a sorting and maximization
                operation, which can be performed on two required and three
                optional input tensors.

**Versioned name**: *NonMaxSuppression-4*

**Category**: *Sorting and maximization*

**Short description**: *NonMaxSuppression* performs non maximum suppression of the boxes with predicted scores.

**Detailed description**: *NonMaxSuppression* performs non maximum suppression algorithm as described below:

1. Take the box with highest score. If the score is less than ``score_threshold`` then stop. Otherwise add the box to the output and continue to the next step.

2. For each input box, calculate the IOU (intersection over union) with the box added during the previous step. If the value is greater than the ``iou_threshold`` threshold then remove the input box from further consideration.

3. Return to step 1.

This algorithm is applied independently to each class of each batch element. The total number of output boxes for each
class must not exceed ``max_output_boxes_per_class``.

**Attributes**:

* *box_encoding*

  * **Description**: *box_encoding* specifies the format of boxes data encoding.
  * **Range of values**: "corner" or "center"

    * *corner* - the box data is supplied as ``[y1, x1, y2, x2]`` where ``(y1, x1)`` and ``(y2, x2)`` are the coordinates of any diagonal pair of box corners.
    * *center* - the box data is supplied as ``[x_center, y_center, width, height]``.
  * **Type**: string
  * **Default value**: "corner"
  * **Required**: *no*

* *sort_result_descending*

  * **Description**: *sort_result_descending* is a flag that specifies whenever it is necessary to sort selected boxes across batches or not.
  * **Range of values**: true of false

    * *true* - sort selected boxes across batches.
    * *false* - do not sort selected boxes across batches (boxes are sorted per class).
  * **Type**: boolean
  * **Default value**: true
  * **Required**: *no*

* *output_type*

  * **Description**: the output tensor type
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i64"
  * **Required**: *no*

**Inputs**:

* **1**: ``boxes`` - tensor of type *T* and shape ``[num_batches, num_boxes, 4]`` with box coordinates. **Required.**

* **2**: ``scores`` - tensor of type *T* and shape ``[num_batches, num_classes, num_boxes]`` with box scores. **Required.**

* **3**: ``max_output_boxes_per_class`` - scalar tensor of type *T_MAX_BOXES* specifying maximum number of boxes to be selected per class. Optional with default value 0 meaning select no boxes.

* **4**: ``iou_threshold`` - scalar tensor of type *T_THRESHOLDS* specifying intersection over union threshold. Optional with default value 0 meaning keep all boxes.

* **5**: ``score_threshold`` - scalar tensor of type *T_THRESHOLDS* specifying minimum score to consider box for the processing. Optional with default value 0.

**Outputs**:

* **1**: ``selected_indices`` - tensor of type *T_IND* and shape ``[min(num_boxes, max_output_boxes_per_class) * num_batches * num_classes, 3]`` containing information about selected boxes as triplets ``[batch_index, class_index, box_index]``.
  The output tensor is filled with -1s for output tensor elements if the total number of selected boxes is less than the output tensor size.

**Types**

* *T*: floating-point type.

* *T_MAX_BOXES*: integer type.

* *T_THRESHOLDS*: floating-point type.

* *T_IND*: ``int64`` or ``int32``.

**Example**

.. code-block::  cpp

  <layer ... type="NonMaxSuppression" ... >
      <data box_encoding="corner" sort_result_descending="1" output_type="i64"/>
      <input>
          <port id="0">
              <dim>3</dim>
              <dim>100</dim>
              <dim>4</dim>
          </port>
          <port id="1">
              <dim>3</dim>
              <dim>5</dim>
              <dim>100</dim>
          </port>
          <port id="2"/> <!-- 10 -->
          <port id="3"/>
          <port id="4"/>
      </input>
      <output>
          <port id="5" precision="I64">
              <dim>150</dim> <!-- min(100, 10) * 3 * 5 -->
              <dim>3</dim>
          </port>
      </output>
  </layer>




