## Sinh <a name="Sinh"></a> {#openvino_docs_ops_arithmetic_Sinh_1}

**Versioned name**: *Sinh-1*

**Category**: Arithmetic unary operation 

**Short description**: *Sinh* performs element-wise hyperbolic sine (sinh) operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise sinh operation. A tensor of type T.

**Types**

* *T*: any numeric type.

*sinh* does the following with the input tensor *a*:

\f[
a_{i} = sinh(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Sinh">
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