## NV12toBGR <a name="NV12toBGR"></a> {#openvino_docs_ops_image_NV12toBGR_8}

**Versioned name**: *NV12toBGR-8*

**Category**: *Image processing*

**Short description**: *NV12toBGR* performs image conversion from NV12 format to BGR.

**Detailed description**:

Similar to *NV12toRGB* but output channels for each pixel are reversed so that: first channel is `blue`, second one is `green`, last one is `red`.  See detailed conversion formulas at [NV12toRGB](NV12toRGB_8.md)

**Inputs:**

Same as specified for [NV12toRGB](NV12toRGB_8.md) operation.

**Outputs:**

* **1**: A tensor of type *T* representing converted image in BGR format. Dimensions:
  * `N` - batch dimension
  * `H` - height dimension is same as image height
  * `W` - width dimension is same as image width
  * `C` - channels dimension equals to 3. First channel is Blue, second one is Green, last one is Red

**Types:**

* *T*: `uint8` or `float32` type.


**Examples:**

*Example 1*

```xml
<layer ... type="NV12toBGR">
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>720</dim>
            <dim>640</dim>
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>480</dim>
            <dim>640</dim>
            <dim>3</dim>
        </port>
    </output>
</layer>
```

*Example 2*

```xml
<layer ... type="NV12toBGR">
    <input>
        <port id="0">  <!-- Y plane -->
            <dim>1</dim>
            <dim>480</dim>
            <dim>640</dim>
            <dim>1</dim>
        </port>
        <port id="1">  <!-- UV plane -->
            <dim>1</dim>
            <dim>240</dim>
            <dim>320</dim>
            <dim>2</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>480</dim>
            <dim>640</dim>
            <dim>3</dim>
        </port>
    </output>
</layer>
```
