## ROIPooling <a name="ROIPooling"></a>

**Versioned name**: *ROIPooling-1*

**Category**: Object detection

**Short description**: *ROIPooling* is a *pooling layer* used over feature maps of non-uniform input sizes and outputs a feature map of a fixed size.

**Detailed description**: [deepsense.io reference](https://blog.deepsense.ai/region-of-interest-pooling-explained/)

**Attributes**

* *pooled_h*

  * **Description**: *pooled_h* is the height of the ROI output feature map. For example, *pooled_h* equal to 6 means that the height of the output of *ROIPooling* is 6.
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* *pooled_w*

  * **Description**: *pooled_w* is the width of the ROI output feature map. For example, *pooled_w* equal to 6 means that the width of the output of *ROIPooling* is 6.
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* *spatial_scale*

  * **Description**: *spatial_scale* is the ratio of the input feature map over the input image size.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* *method*

  * **Description**: *method* specifies a method to perform pooling. If the method is *bilinear*, the input box coordinates are normalized to the `[0, 1]` interval.
  * **Range of values**: *max* or *bilinear*
  * **Type**: string
  * **Default value**: *max*
  * **Required**: *no*

**Inputs**:

*   **1**: 4D input tensor of shape `[1, C, H, W]` with feature maps. Required.

*   **2**: 2D input tensor of shape `[NUM_ROIS, 5]` describing box consisting of 5 element tuples: `[batch_id, x_1, y_1, x_2, y_2]`. Required.

**Outputs**:

*   **1**: 4D output tensor of shape `[NUM_ROIS, C, pooled_h, pooled_w]` with feature maps. Required.

**Example**

```xml
<layer ... type="ROIPooling" ... >
        <data pooled_h="6" pooled_w="6" spatial_scale="0.062500"/>
        <input> ... </input>
        <output> ... </output>
    </layer>
```