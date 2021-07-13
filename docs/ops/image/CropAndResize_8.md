## CropAndResize <a name="CropAndResize"></a> {#openvino_docs_ops_image_CropAndResize_8}

**Versioned name**: *CropAndResize-8*

**Category**: Image processing

**Short description**: *CropAndResize* extracts image patches from input tensor and 
resizes them to a specified output shape.

**Attributes**

* *mode*

    * **Description**: specifies type of interpolation to use when resizing
    * **Range of values**: one of `nearest`, `linear`, `cubic`
    * **Type**: string
    * **Default value**: `nearest`
    * **Required**: *no*

* *extrapolation_value*

    * **Description**: if region of interest falls outside of the original image, 
                       this value is used to extrapolate the missing input image values.
    * **Range of values**: floating point number
    * **Type**: float
    * **Default value**: 0.0f
    * **Required**: *no*
    

* *cube_coeff*

    * **Description**: specifies the parameter *a* for cubic interpolation 
                       (see, e.g.  [article](https://ieeexplore.ieee.org/document/1163711/)).  
                       *cube_coeff* is used only when `mode == cubic`.
    * **Range of values**: floating point number
    * **Type**: any of supported floating point type
    * **Default value**: `-0.75`
    * **Required**: *no*

**Inputs**

*   **1**: `data` - Input tensor with image data in the format `NXYZ...` 
    where `XYZ...` are spatial dimensions of the input image and `N` is batch size.
    Type of elements is any supported floating point type or `int8` type. 
    **Required**.


*   **2**: `boxes` - regions of interest which will be cropped out of the original image and resized.
    
    A 2-D tensor of shape [B, 2*S], where `B` is the number of regions of interest 
    and `S` is the number of spatial dimensions of the `data` tensor. 
    For a 2D image, a single regions's coordinates can be specified as `[y1, x1, y2, x2]`.
    For an S dimensional image, the corrdinates would be 
    `[start_1, start2, ... start_S, end_1, end_2, ... end_S]`.
    
    Coordinates are normalized, so values in the interval `[0, 1]` are
    mapped to `[0, image_dimension - 1]` of the original image.
    
    If `start_S` > `end_S`, the cropped region will be flipped along that axis.
 
    If normalized coordinates fall outside the `[0, 1]` range, 
    `extrapolation_value` is used to fill the missing input image values.
    **Required**.

*   **3**: `box_batch_indices` - 1D tensor of size [B], where `B` is the number of regions of interest.
    Each value is in the range `[0, N)` and specifies which image in the batch the box relates to.
    **Required**.
    
*   **4**: `sizes` - 1D tensor describing output shape for spatial axes. Values: `[out_X, out_Y, out_Z...]`.
    **Required**.


**Outputs**

*   **1**: A N-D tensor of shape [B, out_X, out_Y, out_Z...] 
    containing cropped-out and reshaped regions of interest. 
