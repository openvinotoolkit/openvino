## MaxUnpool <a name="MaxUnpool"></a> {#openvino_docs_ops_unpooling_MaxUnpool}

**Versioned name**: *MaxUnpool*

**Category**: *Unpooling*

**Short description**: Performs max unpooling operation on input.

**Attributes**: *Unooling* has no attributes.

**Inputs**:
  *   **1**: 4D pooling input tensor of shape `[N, C, H, W]` and type *T*. **Required.**
  *   **2**: 4D pooling output tensor of shape `[N, C, H, W]` and type *T*. **Required.**
  *   **3**: 4D input tensor of shape `[N, C, H, W]` and type *T*. **Required.**
  *   **4**: 4D tensor of the required output shape `[N, C, H_out, W_out]` and type *T*. **Required.**

**Outputs**:
  * **1**: The output shape will be `[N, C, H_out, W_out]`. Output tensor has the same data type as input tensor.

**Examples**

```xml
<layer ... type="MaxUnpool" ...>
    <input>
        <port id="0" precision="FP32">
            <dim>5</dim>
            <dim>4</dim>
            <dim>6</dim>
            <dim>9</dim>
        </port>
        <port id="1" precision="FP32">
            <dim>5</dim>
            <dim>4</dim>
            <dim>3</dim>
            <dim>4</dim>
        </port>
        <port id="2" precision="FP32">
            <dim>5</dim>
            <dim>4</dim>
            <dim>3</dim>
            <dim>4</dim>
        </port>
        <port id="3" precision="FP32">
            <dim>5</dim>
            <dim>3</dim>
            <dim>6</dim>
            <dim>9</dim>
        </port>
    </input>
    <output>
        <port id="4" names="output" precision="FP32">
            <dim>5</dim>
            <dim>4</dim>
            <dim>6</dim>
            <dim>9</dim>
        </port>
    </output>
</layer>
```
s