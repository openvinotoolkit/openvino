## Atanh <a name="Atanh"></a> {#openvino_docs_ops_arithmetic_Atanh_3}

**Versioned name**: *Atanh-3*

**Category**: Arithmetic unary operation

**Short description**: *Atanh* performs element-wise hyperbolic inverse tangent (arctangenth) operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: A tensor of type *T*. **Required.**

**Outputs**

* **1**: The result of element-wise atanh operation. A tensor of type *T*.

**Types**

* *T*: any floating-point type.

*Atanh* does the following with the input tensor *a*:

\f[
a_{i} = atanh(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Atanh">
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
