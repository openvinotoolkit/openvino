.. {#openvino_docs_ops_detection_ROIAlignRotated_14}
ROIAlignRotated
===============


.. meta::
  :description: Learn about ROIAlignRotated-14 - an object detection operation, 
                which can be performed on three required input tensors.


**Versioned name**: *ROIAlignRotated-14*

**Category**: *Object detection*

**Short description**: *ROIAlignRotated* is a *pooling layer* used over feature maps of non-uniform input sizes and outputs a feature map of a fixed size.

**Detailed description**: `Reference <https://arxiv.org/abs/1703.06870>`__.

*ROIAlignRotated* performs the following for each Region of Interest (ROI) for each input feature map:

1. Multiply ROI box coordinates with *spatial_scale* to produce box coordinates relative to the input feature map size.
2. Rotate ROI box according to given angle in radians and *clockwise_mode*.
3. Divide the box into equal bins. One bin is mapped to single output feature map element.
4. Inside every bin, calculate regularly spaced sample points, according to the *sampling_ratio* attribute.
5. To calculate the value of single sample point, calculate further 4 points around each sample point to apply bilinear interpolation.
6. Calculate the average of all sample points in the bin to produce output feature map element.

The 4 points used for bilinear interpolation are calculated as the closest integer coordinates to the sample point.
As an example, if the sample point is [2.14, 3.56], then the 4 integer points are [2, 3], [2, 4], [3, 3], [3, 4].

Each ROI box's center is shifted by [-0.5, -0.5] before pooling to achive better alignment with the closest integer coordinates used for bilinear filtering.

**Attributes**

* *pooled_h*

  * **Description**: *pooled_h* is the height of the ROI output feature map.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *pooled_w*

  * **Description**: *pooled_w* is the width of the ROI output feature map.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *sampling_ratio*

  * **Description**: *sampling_ratio* describes the number of sampling points bins over height and width to use to calculate each output feature map element. If the value is greater than 0, then ``bin_points_h = sampling_ratio`` and ``bin_points_w = sampling_ratio``. If the value is equal to 0 then adaptive number of elements over height and width is used: ``bin_points_h = ceil(roi_height / pooled_h)`` and ``bin_points_w = ceil(roi_width / pooled_w)`` respectively. The total number of sampling points for a single bin is equal to ``bin_points_w * bin_points_h``.
  * **Range of values**: a non-negative integer
  * **Type**: ``int``
  * **Required**: *yes*

* *spatial_scale*

  * **Description**: *spatial_scale* is a multiplicative spatial scale factor to that is applied to the ROI box(height, weight and center vector) before pooling.
   WARNING! 
   Spatial scale is also applied to the center point of the ROI box. It means that scaling does not only change the size of the ROI box, but also its position.
   For example, if the spatial scale is 2.0, ROI box center is [0.5, 0.5], box width is 1.0 and box height is 1.0, then after scaling the ROI box center will be [1.0, 1.0], box width will be 2.0 and box height will be 2.0.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

* *clockwise_mode*

  * **Description**:  If True, the angle for each ROI represents a clockwise rotation, otherwise - counterclockwise rotation.
  * **Type**: ``bool``
  * **Default value**: False  
  * **Required**: *no*

**Inputs**:

* **1**: 4D input tensor of shape ``[N, C, H, W]`` with feature maps of type *T*. **Required.**

* **2**: 2D input tensor of shape ``[NUM_ROIS, 5]`` describing ROI box consisting of 5 element tuples: ``[center_x, center_y, width, height, angle]`` in relative coordinates of type *T*. The angle is always in radians.
  * **Required.**

* **3**: 1D input tensor of shape ``[NUM_ROIS]`` with batch indices of type *IND_T*. **Required.**

**Outputs**:

* **1**: 4D output tensor of shape ``[NUM_ROIS, C, pooled_h, pooled_w]`` with feature maps of type *T*.

**Types**

* *T*: any supported floating-point type.

* *IND_T*: any supported integer type.


**Example**

.. code-block:: xml
   :force:

   <layer ... type="ROIAlignRotated" ... >
       <data pooled_h="6" pooled_w="6" spatial_scale="16.0" sampling_ratio="2" clockwise_mode="True"/>
       <input>
           <port id="0">
               <dim>7</dim>
               <dim>256</dim>
               <dim>200</dim>
               <dim>200</dim>
           </port>
           <port id="1">
               <dim>1000</dim>
               <dim>5</dim>
           </port>
           <port id="2">
               <dim>1000</dim>
           </port>
       </input>
       <output>
           <port id="3" precision="FP32">
               <dim>1000</dim>
               <dim>256</dim>
               <dim>6</dim>
               <dim>6</dim>
           </port>
       </output>
   </layer>
