## Asinh <a name="Asinh"></a>

**Versioned name**: *Asinh-4*

**Category**: Arithmetic unary operation 

**Short description**: *Asinh* performs element-wise hyperbolic inverse sine (arcsinh) operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise asinh operation. A tensor of type T.

**Types**

* *T*: any numeric type.

*Asinh* does the following with the input tensor *a*:

\f[
a_{i} = asinh(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Asinh">
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
