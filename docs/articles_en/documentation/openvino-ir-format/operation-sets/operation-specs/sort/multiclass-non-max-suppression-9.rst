MulticlassNonMaxSuppression
===========================

.. meta::
  :description: Learn about MulticlassNonMaxSuppression-8 - a sorting and
                maximization operation, which can be performed on two or three
                required input tensors.

**Versioned name**: *MulticlassNonMaxSuppression-9*

**Category**: *Sorting and maximization*

**Short description**: *MulticlassNonMaxSuppression* performs multi-class non-maximum suppression of the boxes with predicted scores.

**Detailed description**: *MulticlassNonMaxSuppression* is a multi-phase operation. It implements non-maximum suppression algorithm as described below:

1.  Let ``B = [b_0,...,b_n]`` be the list of initial detection boxes, ``S = [s_0,...,s_N]`` be  the list of corresponding scores.
2.  Let ``D = []`` be an initial collection of resulting boxes. Let ``adaptive_threshold = iou_threshold``.
3.  If ``B`` is empty, go to step 9.
4.  Take the box with highest score. Suppose that it is the box ``b`` with the score ``s``.
5.  Delete ``b`` from ``B``.
6.  If the score ``s`` is greater than or equal to ``score_threshold``,  add ``b`` to ``D``, else go to step 9.
7.  If ``nms_eta < 1`` and ``adaptive_threshold > 0.5``, update ``adaptive_threshold *= nms_eta``.
8.  For each input box ``b_i`` from ``B`` and the corresponding score ``s_i``, set ``s_i = 0`` when ``iou(b, b_i) > adaptive_threshold``, and go to step 3.
9.  Return ``D``, a collection of the corresponding scores ``S``, and the number of elements in ``D``.

This algorithm is applied independently to each class of each batch element. The operation feeds at most ``nms_top_k`` scoring candidate boxes to this algorithm.
The total number of output boxes of each batch element must not exceed ``keep_top_k``.
Boxes of ``background_class`` are skipped and thus eliminated.

**Attributes**:

* *sort_result*

  * **Description**: *sort_result* specifies the order of output elements.
  * **Range of values**: ``class``, ``score``, ``none``

    * *class* - sort selected boxes by class id (ascending).
    * *score* - sort selected boxes by score (descending).
    * *none* - do not guarantee the order.

  * **Type**: ``string``
  * **Default value**: ``none``
  * **Required**: *no*

* *sort_result_across_batch*

  * **Description**: *sort_result_across_batch* is a flag that specifies whenever it is necessary to sort selected boxes across batches or not.
  * **Range of values**: true or false

    * *true* - sort selected boxes across batches.
    * *false* - do not sort selected boxes across batches (boxes are sorted per batch element).

  * **Type**: boolean
  * **Default value**: false
  * **Required**: *no*

* *output_type*

  * **Description**: the tensor type of outputs ``selected_indices`` and ``valid_outputs``.
  * **Range of values**: ``i64`` or ``i32``
  * **Type**: ``string``
  * **Default value**: ``i64``
  * **Required**: *no*

* *iou_threshold*

  * **Description**: intersection over union threshold.
  * **Range of values**: a floating-point number
  * **Type**: ``float``
  * **Default value**: ``0``
  * **Required**: *no*

* *score_threshold*

  * **Description**: minimum score to consider box for the processing.
  * **Range of values**: a floating-point number
  * **Type**: ``float``
  * **Default value**: ``0``
  * **Required**: *no*

* *nms_top_k*

  * **Description**: maximum number of boxes to be selected per class.
  * **Range of values**: an integer
  * **Type**: ``int``
  * **Default value**: ``-1`` meaning to keep all boxes
  * **Required**: *no*

* *keep_top_k*

  * **Description**: maximum number of boxes to be selected per batch element.
  * **Range of values**: an integer
  * **Type**: ``int``
  * **Default value**: ``-1`` meaning to keep all boxes
  * **Required**: *no*

* *background_class*

  * **Description**: the background class id.
  * **Range of values**: an integer
  * **Type**: ``int``
  * **Default value**: ``-1`` meaning to keep all classes.
  * **Required**: *no*

* *normalized*

  * **Description**: *normalized* is a flag that indicates whether ``boxes`` are normalized or not.
  * **Range of values**: true or false

    * *true* - the box coordinates are normalized.
    * *false* - the box coordinates are not normalized.

  * **Type**: boolean
  * **Default value**: True
  * **Required**: *no*

* *nms_eta*

  * **Description**: eta parameter for adaptive NMS.
  * **Range of values**: a floating-point number in close range ``[0, 1.0]``.
  * **Type**: ``float``
  * **Default value**: ``1.0``
  * **Required**: *no*

**Inputs**:

There are 2 kinds of input formats. The first one is of two inputs. The boxes are shared by all classes.

* **1**: ``boxes`` - tensor of type *T* and shape ``[num_batches, num_boxes, 4]`` with box coordinates. The box coordinates are layout as ``[xmin, ymin, xmax, ymax]``. **Required.**

* **2**: ``scores`` - tensor of type *T* and shape ``[num_batches, num_classes, num_boxes]`` with box scores. The tensor type should be same with ``boxes``. **Required.**

The second format is of three inputs. Each class has its own boxes that are not shared.
* **1**: ``boxes`` - tensor of type *T* and shape ``[num_classes, num_boxes, 4]`` with box coordinates. The box coordinates are layout as ``[xmin, ymin, xmax, ymax]``. **Required.**

* **2**: ``scores`` - tensor of type *T* and shape ``[num_classes, num_boxes]`` with box scores. The tensor type should be same with ``boxes``. **Required.**

* **3**: ``roisnum`` - tensor of type *T_IND* and shape ``[num_batches]`` with box numbers in each image. ``num_batches`` is the number of images. Each element in this tensor is the number of boxes for corresponding image. The sum of all elements is ``num_boxes``. **Required.**

**Outputs**:

* **1**: ``selected_outputs`` - tensor of type *T* which should be same with ``boxes`` and shape ``[number of selected boxes, 6]`` containing the selected boxes with score and class as tuples ``[class_id, box_score, xmin, ymin, xmax, ymax]``.

* **2**: ``selected_indices`` - tensor of type *T_IND* and shape ``[number of selected boxes, 1]`` the selected indices in the flattened ``boxes``, which are absolute values cross batches. Therefore possible valid values are in the range ``[0, num_batches * num_boxes - 1]``.

* **3**: ``selected_num`` - 1D tensor of type *T_IND* and shape ``[num_batches]`` representing the number of selected boxes for each batch element.

When there is no box selected, ``selected_num`` is filled with ``0``. ``selected_outputs`` is an empty tensor of shape ``[0, 6]``, and ``selected_indices`` is an empty tensor of shape ``[0, 1]``.

**Types**

* *T*: floating-point type.

* *T_IND*: ``int64`` or ``int32``.

**Example**

.. code-block:: cpp

   <layer ... type="MulticlassNonMaxSuppression" ... >
       <data sort_result="score" output_type="i64" sort_result_across_batch="false" iou_threshold="0.2" score_threshold="0.5" nms_top_k="-1" keep_top_k="-1" background_class="-1"    normalized="false" nms_eta="0.0"/>
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
       </input>
       <output>
           <port id="5" precision="FP32">
               <dim>-1</dim> <!-- "-1" means a undefined dimension calculated during the model inference -->
               <dim>6</dim>
           </port>
           <port id="6" precision="I64">
               <dim>-1</dim>
               <dim>1</dim>
           </port>
           <port id="7" precision="I64">
               <dim>3</dim>
           </port>
       </output>
   </layer>


Another possible example with 3 inputs could be like:


.. code-block:: cpp

   <layer ... type="MulticlassNonMaxSuppression" ... >
       <data sort_result="score" output_type="i64" sort_result_across_batch="false" iou_threshold="0.2" score_threshold="0.5" nms_top_k="-1" keep_top_k="-1" background_class="-1"    normalized="false" nms_eta="0.0"/>
       <input>
           <port id="0">
               <dim>3</dim>
               <dim>100</dim>
               <dim>4</dim>
           </port>
           <port id="1">
               <dim>3</dim>
               <dim>100</dim>
           </port>
           <port id="2">
               <dim>10</dim>
           </port>
       </input>
       <output>
           <port id="5" precision="FP32">
               <dim>-1</dim> <!-- "-1" means a undefined dimension calculated during the model inference -->
               <dim>6</dim>
           </port>
           <port id="6" precision="I64">
               <dim>-1</dim>
               <dim>1</dim>
           </port>
           <port id="7" precision="I64">
               <dim>3</dim>
           </port>
       </output>
   </layer>



