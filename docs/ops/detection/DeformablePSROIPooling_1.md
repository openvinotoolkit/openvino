## DeformablePSROIPooling <a name="DeformablePSROIPooling"></a> {#openvino_docs_ops_detection_DeformablePSROIPooling_1}

**Versioned name**: *DeformablePSROIPooling-1*

**Category**: Object detection

**Short description**: *DeformablePSROIPooling* computes position-sensitive pooling on regions of interest specified by input.

**Detailed description**: [Reference](https://arxiv.org/abs/1703.06211).

*DeformablePSROIPooling* operation takes two or three input tensors: with feature maps, with regions of interests (box coordinates) and an optional tensor with transformation values.
The box coordinates are specified as five element tuples: *[batch_id, x_1, y_1, x_2, y_2]* in absolute values.

**Attributes**

* *output_dim*

  * **Description**: *output_dim* is a pooled output channel number.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* *group_size*

  * **Description**: *group_size* is the number of groups to encode position-sensitive score maps.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* *spatial_scale*

  * **Description**: *spatial_scale* is a multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* *mode*
  * **Description**: *mode* specifies mode for pooling.
  * **Range of values**:
    * *bilinear_deformable* - perform pooling with bilinear interpolation and deformable transformation
  * **Type**: string
  * **Default value**: *bilinear_deformable*
  * **Required**: *no*

* *spatial_bins_x*
  * **Description**: *spatial_bins_x* specifies numbers of bins to divide the input feature maps over width.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* *spatial_bins_y*
  * **Description**: *spatial_bins_y* specifies numbers of bins to divide the input feature maps over height.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* *trans_std*
  * **Description**: *trans_std* is the value that all transformation (offset) values are multiplied with.
  * **Range of values**: floating point number
  * **Type**: `float`
  * **Default value**: 1
  * **Required**: *no*

* *part_size*
  * **Description**: *part_size* is the number of parts the output tensor spatial dimensions are divided into. Basically it is the height and width of the third input 
  with transformation values.
  * **Range of values**: positive integer number
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

*   **1**: 4D input tensor with feature maps. Required.

*   **2**: 2D input tensor describing box consisting of five element tuples: `[batch_id, x_1, y_1, x_2, y_2]`. Required.

*   **3**: 4D input blob with transformation [values](https://arxiv.org/abs/1703.06211) (offsets). Optional.

**Outputs**:

*   **1**: 4D output tensor with areas copied and interpolated from the 1st input tensor by coordinates of boxes from the 2nd input and transformed according to values from the 3rd input.

**Example**

```xml
<layer ... type="DeformablePSROIPooling" ... >
    <data group_size="7" mode="bilinear_deformable" output_dim="8" part_size="7" spatial_bins_x="4" spatial_bins_y="4" spatial_scale="0.0625" trans_std="0.1"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>392</dim>
            <dim>38</dim>
            <dim>63</dim>
        </port>
        <port id="1">
            <dim>300</dim>
            <dim>5</dim>
        </port>
        <port id="2">
            <dim>300</dim>
            <dim>2</dim>
            <dim>7</dim>
            <dim>7</dim>
        </port>
    </input>
    <output>
        <port id="3" precision="FP32">
            <dim>300</dim>
            <dim>8</dim>
            <dim>7</dim>
            <dim>7</dim>
        </port>
    </output>
</layer>
```
