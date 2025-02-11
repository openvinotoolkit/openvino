ROIPooling
==========


.. meta::
  :description: Learn about ROIPooling-1 - an object detection operation,
                which can be performed on two required input tensors.

**Versioned name**: *ROIPooling-1*

**Category**: *Object detection*

**Short description**: *ROIPooling* is a *pooling layer* used over feature maps of non-uniform input sizes and outputs a feature map of a fixed size.

**Detailed description**:

*ROIPooling* performs the following operations for each Region of Interest (ROI) over the input feature maps:

1. Produce box coordinates relative to the input feature map size, based on *method* attribute.
2. Calculate box height and width.
3. Divide the box into bins according to the pooled size attributes, ``[pooled_h, pooled_w]``.
4. Apply maximum or bilinear interpolation pooling, for each bin, based on *method* attribute to produce output feature map element.

The box height and width have different representation based on **method** attribute:

* *max*: Expressed in relative coordinates. The box height and width are calculated the following way: ``roi_width = max(spatial_scale * (x_2 - x_1), 1.0)``, ``roi_height = max(spatial_scale * (y_2 - y_1), 1.0)``, so the malformed boxes are expressed as a box of size ``1 x 1``.
* *bilinear*: Expressed in absolute coordinates and normalized to the ``[0, 1]`` interval. The box height and width are calculated the following way: ``roi_width = (W - 1)  * (x_2 - x_1)``, ``roi_height = (H - 1) * (y_2 - y_1)``.

**Attributes**

* *pooled_h*

  * **Description**: *pooled_h* is the height of the ROI output feature map. For example, *pooled_h* equal to 6 means that the height of the output of *ROIPooling* is 6.
  * **Range of values**: a non-negative integer
  * **Type**: ``int``
  * **Required**: *yes*

* *pooled_w*

  * **Description**: *pooled_w* is the width of the ROI output feature map. For example, *pooled_w* equal to 6 means that the width of the output of *ROIPooling* is 6.
  * **Range of values**: a non-negative integer
  * **Type**: ``int``
  * **Required**: *yes*

* *spatial_scale*

  * **Description**: *spatial_scale* is the ratio of the input feature map over the input image size.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

* *method*

  * **Description**: *method* specifies a method to perform pooling. If the method is *bilinear*, the input box coordinates are normalized to the ``[0, 1]`` interval.
  * **Range of values**: *max* or *bilinear*
  * **Type**: string
  * **Default value**: *max*
  * **Required**: *no*

**Inputs**:

* **1**: 4D input tensor of shape ``[N, C, H, W]`` with feature maps of type *T*. **Required.**

* **2**: 2D input tensor of shape ``[NUM_ROIS, 5]`` describing region of interest box consisting of 5 element tuples of type *T*: ``[batch_id, x_1, y_1, x_2, y_2]``. **Required.**
  Batch indices must be in the range of ``[0, N-1]``.


**Outputs**:

*   **1**: 4D output tensor of shape ``[NUM_ROIS, C, pooled_h, pooled_w]`` with feature maps of type *T*.

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="ROIPooling" ... >
           <data pooled_h="6" pooled_w="6" spatial_scale="0.062500"/>
           <input> ... </input>
           <output> ... </output>
       </layer>

