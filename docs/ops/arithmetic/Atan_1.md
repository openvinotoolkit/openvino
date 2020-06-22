## Atan <a name="Atan"></a>

**Versioned name**: *Atan-1*

**Category**: Arithmetic unary operation 

**Short description**: *Atan* performs element-wise inverse tangent (arctangent) operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise atan operation. A tensor of type T.

**Types**

* *T*: any numeric type.

*atan* does the following with the input tensor *a*:

\f[
a_{i} = atan(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Atan">
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
