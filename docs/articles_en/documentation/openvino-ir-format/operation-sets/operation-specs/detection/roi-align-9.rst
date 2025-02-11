ROIAlign
========


.. meta::
  :description: Learn about ROIAlign-9 - an object detection operation,
                which can be performed on three required input tensors.


**Versioned name**: *ROIAlign-9*

**Category**: *Object detection*

**Short description**: *ROIAlign* is a *pooling layer* used over feature maps of non-uniform input sizes and outputs a feature map of a fixed size.

**Detailed description**: `Reference <https://arxiv.org/abs/1703.06870>`__.

*ROIAlign* performs the following for each Region of Interest (ROI) for each input feature map:

1. Multiply box coordinates with *spatial_scale* to produce box coordinates relative to the input feature map size based on *aligned_mode* attribute.
2. Divide the box into bins according to the *sampling_ratio* attribute.
3. Apply bilinear interpolation with 4 points in each bin and apply maximum or average pooling based on *mode* attribute to produce output feature map element.

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

  * **Description**: *sampling_ratio* is the number of bins over height and width to use to calculate each output feature map element. If the value is equal to 0 then use adaptive number of elements over height and width: ``ceil(roi_height / pooled_h)`` and ``ceil(roi_width / pooled_w)`` respectively.
  * **Range of values**: a non-negative integer
  * **Type**: ``int``
  * **Required**: *yes*

* *spatial_scale*

  * **Description**: *spatial_scale* is a multiplicative spatial scale factor to translate ROI coordinates from their input spatial scale to the scale used when pooling.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

* *mode*

  * **Description**: *mode* specifies a method to perform pooling to produce output feature map elements.
  * **Range of values**:

    * *max* - maximum pooling
    * *avg* - average pooling
  * **Type**: string
  * **Required**: *yes*

* *aligned_mode*

  * **Description**: *aligned_mode* specifies how to transform the coordinate in original tensor to the resized tensor.
  * **Range of values**: name of the transformation mode in string format (here spatial_scale is resized_shape[x] / original_shape[x], resized_shape[x] is the shape of resized tensor in axis x, original_shape[x] is the shape of original tensor in axis x and x_original is a coordinate in axis x, for any axis x from the input axes):

    * *asymmetric* - the coordinate in the resized tensor axis x is calculated according to the formula x_original * spatial_scale
    * *half_pixel_for_nn* - the coordinate in the resized tensor axis x is x_original * spatial_scale - 0.5
    * *half_pixel* - the coordinate in the resized tensor axis x is calculated as ((x_original + 0.5) * spatial_scale) - 0.5
  * **Type**: string
  * **Default value**: asymmetric
  * **Required**: *no*

**Inputs**:

* **1**: 4D input tensor of shape ``[N, C, H, W]`` with feature maps of type *T*. **Required.**

* **2**: 2D input tensor of shape ``[NUM_ROIS, 4]`` describing box consisting of 4 element tuples: ``[x_1, y_1, x_2, y_2]`` in relative coordinates of type *T*. The box height and width are calculated the following way:

  * If *aligned_mode* equals *asymmetric*: ``roi_width = max(spatial_scale * (x_2 - x_1), 1.0)``, ``roi_height = max(spatial_scale * (y_2 - y_1), 1.0)``, so the malformed boxes are expressed as a box of size ``1 x 1``.
  * else: ``roi_width = spatial_scale * (x_2 - x_1)``, ``roi_height = spatial_scale * (y_2 - y_1)``.
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

   <layer ... type="ROIAlign" ... >
       <data pooled_h="6" pooled_w="6" spatial_scale="16.0" sampling_ratio="2" mode="avg" aligned_mode="half_pixel"/>
       <input>
           <port id="0">
               <dim>7</dim>
               <dim>256</dim>
               <dim>200</dim>
               <dim>200</dim>
           </port>
           <port id="1">
               <dim>1000</dim>
               <dim>4</dim>
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

