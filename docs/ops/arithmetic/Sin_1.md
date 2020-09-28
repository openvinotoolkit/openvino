## Sin <a name="Sin"></a> {#openvino_docs_ops_arithmetic_Sin_1}

**Versioned name**: *Sin-1*

**Category**: Arithmetic unary operation 

**Short description**: *Sin* performs element-wise sine operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise sin operation. A tensor of type T.

**Types**

* *T*: any numeric type.

*sin* does the following with the input tensor *a*:

\f[
a_{i} = sin(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Sin">
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