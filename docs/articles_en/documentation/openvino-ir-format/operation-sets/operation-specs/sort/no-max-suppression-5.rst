NonMaxSuppression
=================


.. meta::
  :description: Learn about NonMaxSuppression-5 - a sorting and maximization
                operation, which can be performed on two required and four
                optional input tensors.

**Versioned name**: *NonMaxSuppression-5*

**Category**: *Sorting and maximization*

**Short description**: *NonMaxSuppression* performs non maximum suppression of the boxes with predicted scores.

**Detailed description**: *NonMaxSuppression* performs non maximum suppression algorithm as described below:

1.  Let ``B = [b_0,...,b_n]`` be the list of initial detection boxes, ``S = [s_0,...,s_N]`` be  the list of corresponding scores.
2.  Let ``D = []`` be an initial collection of resulting boxes.
3.  If ``B`` is empty then go to step 8.
4.  Take the box with highest score. Suppose that it is the box ``b`` with the score ``s``.
5.  Delete ``b`` from ``B``.
6.  If the score ``s`` is greater or equal than ``score_threshold``  then add ``b`` to ``D`` else go to step 8.
7.  For each input box ``b_i`` from ``B`` and the corresponding score ``s_i``, set ``s_i = s_i * func(IOU(b_i, b))`` and go to step 3.
8.  Return ``D``, a collection of the corresponding scores ``S``, and the number of elements in ``D``.

Here ``func(iou) = 1 if iou <= iou_threshold else 0`` when ``soft_nms_sigma == 0``, else ``func(iou) = exp(-0.5 * iou * iou / soft_nms_sigma) if iou <= iou_threshold else 0``.

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

*   **1**: ``boxes`` - tensor of type *T* and shape ``[num_batches, num_boxes, 4]`` with box coordinates. **Required.**

*   **2**: ``scores`` - tensor of type *T* and shape ``[num_batches, num_classes, num_boxes]`` with box scores. **Required.**

*   **3**: ``max_output_boxes_per_class`` - scalar or 1D tensor with 1 element of type *T_MAX_BOXES* specifying maximum number of boxes to be selected per class. Optional with default value 0 meaning select no boxes.

*   **4**: ``iou_threshold`` - scalar or 1D tensor with 1 element of type *T_THRESHOLDS* specifying intersection over union threshold. Optional with default value 0 meaning keep all boxes.

*   **5**: ``score_threshold`` - scalar or 1D tensor with 1 element of type *T_THRESHOLDS* specifying minimum score to consider box for the processing. Optional with default value 0.

*   **6**:  ``soft_nms_sigma`` - scalar or 1D tensor with 1 element of type *T_THRESHOLDS* specifying the sigma parameter for Soft-NMS; see `Bodla et al <https://arxiv.org/abs/1704.04503.pdf>`__. Optional with default value 0.

**Outputs**:

*   **1**: ``selected_indices`` - tensor of type *T_IND* and shape ``[number of selected boxes, 3]`` containing information about selected boxes as triplets ``[batch_index, class_index, box_index]``.

*   **2**: ``selected_scores`` - tensor of type *T_THRESHOLDS* and shape ``[number of selected boxes, 3]`` containing information about scores for each selected box as triplets ``[batch_index, class_index, box_score]``.

*   **3**: ``valid_outputs`` - 1D tensor with 1 element of type *T_IND* representing the total number of selected boxes.

Plugins which do not support dynamic output tensors produce ``selected_indices`` and ``selected_scores`` tensors of shape ``[min(num_boxes, max_output_boxes_per_class) * num_batches * num_classes, 3]`` which is an upper bound for the number of possible selected boxes. Output tensor elements following the really selected boxes are filled with value -1.

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
          <port id="6" precision="FP32">
              <dim>150</dim> <!-- min(100, 10) * 3 * 5 -->
              <dim>3</dim>
          </port>
          <port id="7" precision="I64">
              <dim>1</dim>
          </port>
      </output>
  </layer>



