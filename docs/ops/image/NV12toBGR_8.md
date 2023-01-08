# NV12toBGR {#openvino_docs_ops_image_NV12toBGR_8}

**Versioned name**: *NV12toBGR-8*

**Category**: *Image processing*

**Short description**: *NV12toBGR* performs image conversion from NV12 to BGR format.

**Detailed description**:

Similar to *NV12toRGB* but output channels for each pixel are reversed so that the first channel is `blue`, the second one is `green`, the last one is `red`.  See detailed conversion formulas in the [NV12toRGB description](NV12toRGB_8.md).

**Inputs:**

Same as specified for [NV12toRGB](NV12toRGB_8.md) operation.

**Outputs:**

* **1**: A tensor of type *T* representing an image converted in BGR format. Dimensions:
  * `N` - batch dimension
  * `H` - height dimension is the same as the image height
  * `W` - width dimension is the same as the image width
  * `C` - channels dimension is equal to 3. The first channel is Blue, the second one is Green, the last one is Red

**Types:**

* *T*: `uint8` or any supported floating-point type.


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
