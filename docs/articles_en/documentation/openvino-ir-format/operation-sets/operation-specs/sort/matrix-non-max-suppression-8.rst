MatrixNonMaxSuppression
=======================


.. meta::
  :description: Learn about MatrixNonMaxSuppression-8 - a sorting and
                maximization operation, which can be performed on two required
                input tensors.

**Versioned name**: *MatrixNonMaxSuppression-8*

**Category**: *Sorting and maximization*

**Short description**: *MatrixNonMaxSuppression* performs matrix non-maximum suppression (NMS) of the boxes with predicted scores.

**Detailed description**: The operation performs the following:

1. Selects candidate bounding boxes with scores higher than ``score_threshold``.
2. For each class, selects at most ``nms_top_k`` candidate boxes.
3. Decays scores of the candidate boxes according to the Matrix NMS algorithm `Wang et al <https://arxiv.org/abs/2003.10152.pdf>`__. This algorithm is applied independently to each class and each batch element. Boxes of ``background_class`` are skipped and thus eliminated during the process.
4. Selects boxes with the decayed scores higher than ``post_threshold``, and selects at most ``keep_top_k`` scoring candidate boxes per batch element.

The Matrix NMS algorithm is described below:

1.  Sort descending the candidate boxes by score, and compute ``n*n`` pairwise IOU (IntersectionOverUnion) matrix ``X`` for the top ``n`` boxes. Suppose ``n`` is the number of candidate boxes.
2.  Set the lower triangle and diagonal of ``X`` to 0. Therefore get the upper triangular matrix ``X``.
3.  Take the column-wise max of ``X`` to compute a vector ``K`` of maximum IOU for each candidate box.
4.  Repeat element value of ``K`` along axis 1. Suppose this gets a matrix ``X_cmax``.
5.  Compute the decay factor: ``decay_factor = exp((X_cmax**2 - X**2) * gaussian_sigma)`` if ``decay_function`` is ``gaussian``, else ``decay_factor = (1 - X) / (1 - X_cmax)``.
6.  Take the column-wise min of ``decay_factor``, and element-wise multiply with scores to decay them.

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
  * **Default value**: ``-1`` meaning to keep all classes
  * **Required**: *no*

* *normalized*

  * **Description**: *normalized* is a flag that indicates whether ``boxes`` are normalized or not.
  * **Range of values**: true or false

    * *true* - the box coordinates are normalized.
    * *false* - the box coordinates are not normalized.

  * **Type**: boolean
  * **Default value**: True
  * **Required**: *no*

* *decay_function*

  * **Description**: decay function used to decay scores.
  * **Range of values**: ``gaussian``, ``linear``
  * **Type**: ``string``
  * **Default value**: ``linear``
  * **Required**: *no*

* *gaussian_sigma*

  * **Description**: gaussian_sigma parameter for gaussian decay_function.
  * **Range of values**: a floating-point number
  * **Type**: ``float``
  * **Default value**: ``2.0``
  * **Required**: *no*

* *post_threshold*

  * **Description**: threshold to filter out boxes with low confidence score after decaying.
  * **Range of values**: a floating-point number
  * **Type**: ``float``
  * **Default value**: ``0``
  * **Required**: *no*

**Inputs**:

* **1**: ``boxes`` - tensor of type *T* and shape ``[num_batches, num_boxes, 4]`` with box coordinates. The box coordinates are layout as ``[xmin, ymin, xmax, ymax]``. **Required.**

* **2**: ``scores`` - tensor of type *T* and shape ``[num_batches, num_classes, num_boxes]`` with box scores. The tensor type should be same with ``boxes``. **Required.**

**Outputs**:

* **1**: ``selected_outputs`` - tensor of type *T* which should be same with ``boxes`` and shape ``[number of selected boxes, 6]`` containing the selected boxes with score and class as tuples ``[class_id, box_score, xmin, ymin, xmax, ymax]``.

* **2**: ``selected_indices`` - tensor of type *T_IND* and shape ``[number of selected boxes, 1]`` the selected indices in the flattened input ``boxes``, which are absolute values cross batches. Therefore possible valid values are in the range ``[0, num_batches * num_boxes - 1]``.

* **3**: ``selected_num`` - 1D tensor of type *T_IND* and shape ``[num_batches]`` representing the number of selected boxes for each batch element.

When there is no box selected, ``selected_num`` is filled with ``0``. ``selected_outputs`` is an empty tensor of shape ``[0, 6]``, and ``selected_indices`` is an empty tensor of shape ``[0, 1]``.

**Types**

* *T*: floating-point type.

* *T_IND*: ``int64`` or ``int32``.

**Example**

.. code-block:: cpp

   <layer ... type="MatrixNonMaxSuppression" ... >
       <data decay_function="gaussian" sort_result="score" output_type="i64"/>
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



