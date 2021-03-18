## ExperimentalDetectronPriorGridGenerator <a name="ExperimentalDetectronPriorGridGenerator"></a> {#openvino_docs_ops_detection_ExperimentalDetectronPriorGridGenerator_6}

**Versioned name**: *ExperimentalDetectronPriorGridGenerator-6*

**Category**: Object detection

**Short description**: An operation *ExperimentalDetectronPriorGridGenerator* operation generates prior grids of specified sizes.

**Detailed description**: Operation takes coordinates of centres of boxes and add strides to them to calculate coordinates of prior grids according to next algorithm:
    
    for (int h = 0; h < layer_height; ++h)
        for (int w = 0; w < layer_width; ++w)
            for (int s = 0; s < number_of_priors; ++s)
                data[0] = priors[4 * s + 0] + step_w * (w + 0.5)
                data[1] = priors[4 * s + 1] + step_h * (h + 0.5)
                data[2] = priors[4 * s + 2] + step_w * (w + 0.5)
                data[3] = priors[4 * s + 3] + step_h * (h + 0.5)
                data += 4;

If *h* and *w* are zeroes, then `layer_height` = `featmap_height` and `layer_width` = `featmap_width`, otherwise *h* and *w* respectively.

If *stride_h* and *stride_w* are zeroes then `step_h` = `image_height` / `layer_height` and `step_w` = `image_width` / `layer_width`, otherwise *stride_h* and *stride_w* respectively.

`featmap_height`, `featmap_width`, `image_height` and `image_width` are spatial dimensions values from second and third inputs respectively.

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
    * **Range of values**: non-negative integer number less than `featmap_height`
    * **Type**: int
    * **Default value**: 0
    * **Required**: *no*
    
* *w*

    * **Description**: *w* attribute specifies number of cells of the generated grid with respect to width.
    * **Range of values**: non-negative integer number less than `featmap_width`
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

* **1**: A tensor of type *T* with priors. Rank must be equal to 2 and The last dimension must be equal to 4: `[number_of_priors, 4]`. **Required.**

* **2**: A 4D tensor of type *T* with input feature map. **Required.**

* **3**: A 4D tensor of type *T* with input image. The number of channels of both feature map and input image tensors must match. **Required.**

**Outputs**

* **1**: A tensor of type *T* with priors grid with shape `[featmap_height * featmap_width * number_of_priors, 4]` if flatten is `true` or `[featmap_height, featmap_width, number_of_priors, 4]` otherwise.
In case then 0 < *h* < `featmap_height` and/or 0 < *w* < `featmap_width` the output data size is less than `featmap_height` * `featmap_width` * `number_of_priors` * 4.
The output tensor is filled with -1s ??? for output tensor elements if the output data size is less than the output tensor size.

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
