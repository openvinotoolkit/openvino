# I420toBGR {#openvino_docs_ops_image_I420toBGR_8}

**Versioned name**: *I420toBGR-8*

**Category**: *Image processing*

**Short description**: *I420toBGR* performs image conversion from I420 to BGR format.

**Detailed description**:

Similar to *I420toRGB* but output channels for each pixel are reversed so that the first channel is `blue`, the second one is `green`, the last one is `red`.  See detailed conversion formulas in the [I420toRGB description](I420toRGB_8.md).

**Inputs:**

Same as specified for [I420toRGB](I420toRGB_8.md) operation.

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
<layer ... type="I420toBGR">
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
<layer ... type="I420toBGR">
    <input>
        <port id="0">  <!-- Y plane -->
            <dim>1</dim>
            <dim>480</dim>
            <dim>640</dim>
            <dim>1</dim>
        </port>
        <port id="1">  <!-- U plane -->
            <dim>1</dim>
            <dim>240</dim>
            <dim>320</dim>
            <dim>1</dim>
        </port>
        <port id="2">  <!-- V plane -->
          <dim>1</dim>
          <dim>240</dim>
          <dim>320</dim>
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
