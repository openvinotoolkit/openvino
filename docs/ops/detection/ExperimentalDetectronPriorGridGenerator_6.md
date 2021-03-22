## ExperimentalDetectronPriorGridGenerator <a name="ExperimentalDetectronPriorGridGenerator"></a> {#openvino_docs_ops_detection_ExperimentalDetectronPriorGridGenerator_6}

**Versioned name**: *ExperimentalDetectronPriorGridGenerator-6*

**Category**: Object detection

**Short description**: An operation *ExperimentalDetectronPriorGridGenerator* operation generates prior grids of 
specified sizes.

**Detailed description**: Operation takes coordinates of centres of boxes and add strides to them to calculate 
coordinates of prior grids according to next algorithm:
    
    for (int ih = 0; ih < layer_height; ++ih)
        for (int iw = 0; iw < layer_width; ++iw)
            for (int s = 0; s < number_of_priors; ++s)
                output_data[0] = priors[4 * s + 0] + step_w * (iw + 0.5)
                output_data[1] = priors[4 * s + 1] + step_h * (ih + 0.5)
                output_data[2] = priors[4 * s + 2] + step_w * (iw + 0.5)
                output_data[3] = priors[4 * s + 3] + step_h * (ih + 0.5)
                output_data += 4

`featmap_height`, `featmap_width`, `image_height` and `image_width` are spatial dimensions values from second and third 
inputs respectively. `priors` is a data from first input.

If *h* and *w* are zeroes, then `layer_height` = `featmap_height` and `layer_width` = `featmap_width`, otherwise *h* 
and *w* respectively.

If *stride_h* and *stride_w* are zeroes then `step_h` = `image_height` / `layer_height` and 
`step_w` = `image_width` / `layer_width`, otherwise *stride_h* and *stride_w* respectively.

**Attributes**:

* *flatten*

    * **Description**: *flatten* attribute specifies whether the output tensor should be 2D or 4D.
    * **Range of values**:
      * `true` - the output tensor should be 2D tensor
      * `false` - the output tensor should be 4D tensor
    * **Type**: boolean
    * **Default value**: true
    * **Required**: *no*

* *h*

    * **Description**: *h* attribute specifies number of cells of the generated grid with respect to height.
    * **Range of values**: non-negative integer number less or equal than `featmap_height`
    * **Type**: int
    * **Default value**: 0
    * **Required**: *no*
    
* *w*

    * **Description**: *w* attribute specifies number of cells of the generated grid with respect to width.
    * **Range of values**: non-negative integer number less or equal than `featmap_width`
    * **Type**: int
    * **Default value**: 0
    * **Required**: *no*

* *stride_x*

    * **Description**: *stride_x* attribute specifies the step of generated grid with respect to x coordinate.
    * **Range of values**: non-negative float number
    * **Type**: float
    * **Default value**: 0.0
    * **Required**: *no*
    
* *stride_y*

    * **Description**: *stride_y* attribute specifies the step of generated grid with respect to y coordinate.
    * **Range of values**: non-negative float number
    * **Type**: float
    * **Default value**: 0.0
    * **Required**: *no*

**Inputs**

* **1**: A 2D tensor of type *T* with shape `[number_of_priors, 4]` contains priors. **Required.**

* **2**: A 4D tensor of type *T* with input feature map `[1, number_of_channels, featmap_height, featmap_width]`. This 
operation uses only sizes of this input tensor, not its data.**Required.**

* **3**: A 4D tensor of type *T* with input image `[1, number_of_channels, image_height, image_width]`. The number of 
channels of both feature map and input image tensors must match. This operation uses only sizes of this input tensor, 
not its data. **Required.**

**Outputs**

* **1**: A tensor of type *T* with priors grid with shape `[featmap_height * featmap_width * number_of_priors, 4]` 
if flatten is `true` or `[featmap_height, featmap_width, number_of_priors, 4]` otherwise.
In case then 0 < *h* < `featmap_height` and/or 0 < *w* < `featmap_width` the output data size is less than 
`featmap_height` * `featmap_width` * `number_of_priors` * 4 and the output tensor is filled with undefined values for 
rest output tensor elements.

**Types**

* *T*: any supported floating point type.

**Example**

```xml
<layer ... type="ExperimentalDetectronPriorGridGenerator" version="opset6">
    <data flatten="true" h="0" stride_x="32.0" stride_y="32.0" w="0"/>
    <input>
        <port id="0">
            <dim>3</dim>
            <dim>4</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>256</dim>
            <dim>25</dim>
            <dim>42</dim>
        </port>
        <port id="2">
            <dim>1</dim>
            <dim>3</dim>
            <dim>800</dim>
            <dim>1344</dim>
        </port>
    </input>
    <output>
        <port id="3" precision="FP32">
            <dim>3150</dim>
            <dim>4</dim>
        </port>
    </output>
</layer>
```
