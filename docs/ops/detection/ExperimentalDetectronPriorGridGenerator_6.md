## ExperimentalDetectronPriorGridGenerator <a name="ExperimentalDetectronPriorGridGenerator"></a> {#openvino_docs_ops_detection_ExperimentalDetectronPriorGridGenerator_6}

**Versioned name**: *ExperimentalDetectronPriorGridGenerator-6*

**Category**: Object detection

**Short description**: An operation *ExperimentalDetectronPriorGridGenerator* operation generates prior grids of specified sizes.

**Detailed description**: TBD

**Attributes**:

* *flatten*

    * **Description**: *flatten* attribute specifies whether the output tensor should be 2D or 4D.
    * **Range of values**:
      * `true` - the output tensor should be 2D tensor
      * `false` - the output tensor should be 4D tensor
    * **Type**: boolean
    * **Default value**: true
    * **Required**: *yes*

* *h* (*w*)

    * **Description**: *h* (*w*) attribute specifies number of cells of the generated grid with respect to height (width).
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: 0
    * **Required**: *yes*

* *stride_x* (*stride_y*)

    * **Description**: *stride_x* (*stride_y*) attribute specifies the step of generated grid with respect to x (y) coordinate.
    * **Range of values**: non-negative float number
    * **Type**: float
    * **Default value**: 0.0
    * **Required**: *yes*

**Inputs**

* **1**: A tensor of type *T* with priors. Rank must be equal to 2 and The last dimension must be equal to 4: `[number_of_priors, 4]`. **Required.**

* **2**: A 4D tensor of type *T* with input feature map. **Required.**

* **3**: A 4D tensor of type *T* with input data. The number of channels of both feature map and input data tensors must match. **Required.**

**Outputs**

* **1**: A tensor of type *T* with priors grid with shape `[featmap_height * featmap_width * number_of_priors, 4]` if flatten is `true` or `[featmap_height, featmap_width, number_of_priors, 4]` otherwise, where `featmap_height` and `featmap_width` are spatial dimensions values from second input.

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
