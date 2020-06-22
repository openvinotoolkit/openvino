## Atanh <a name="Atanh"></a>

**Versioned name**: *Atanh-1*

**Category**: Arithmetic unary operation 

**Short description**: *Atanh* performs element-wise hyperbolic inverse tangent (arctangenth) operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise atanh operation. A tensor of type T.

**Types**

* *T*: any numeric type.

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
