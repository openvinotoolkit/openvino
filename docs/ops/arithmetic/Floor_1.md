## Floor <a name="Floor"></a> {#openvino_docs_ops_arithmetic_Floor_1}

**Versioned name**: *Floor-1*

**Category**: Arithmetic unary operation 

**Short description**: *Floor* performs element-wise floor operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise floor operation. A tensor of type T.

**Types**

* *T*: any numeric type.

*Floor* does the following with the input tensor *a*:

\f[
a_{i} = floor(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Floor">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```
