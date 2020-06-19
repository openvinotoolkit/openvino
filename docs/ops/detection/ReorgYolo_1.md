## ReorgYolo Layer <a name="ReorgYolo"></a>

**Versioned name**: *ReorgYolo-1*

**Category**: *Object detection*

**Short description**: *ReorgYolo* reorganizes input tensor taking into account strides.

**Detailed description**:

[Reference](https://arxiv.org/pdf/1612.08242.pdf)

**Attributes**

* *stride*

  * **Description**: *stride* is the distance between cut throws in output blobs.
  * **Range of values**: positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: 4D input tensor of any type and shape `[N, C, H, W]`. `H` and `W` should be divisible by `stride`. Required.

**Outputs**:

*   **1**: 4D output tensor of the same type as input tensor and shape `[N, C*stride*stride, H/stride, W/stride]`. Required.

**Example**

```xml
<layer id="89" name="ExtractImagePatches" type="ReorgYolo">
    <data stride="2"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>64</dim>
            <dim>26</dim>
            <dim>26</dim>
        </port>
    </input>
    <output>
        <port id="1" precision="f32">
            <dim>1</dim>
            <dim>256</dim>
            <dim>13</dim>
            <dim>13</dim>
        </port>
    </output>
</layer>
```