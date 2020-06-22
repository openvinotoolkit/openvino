## Ceiling <a name="Ceiling"></a>

**Versioned name**: *Ceiling-1*

**Category**: Arithmetic unary operation 

**Short description**: *Ceiling* performs element-wise ceiling operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise ceiling operation. A tensor of type T.

**Types**

* *T*: any numeric type.

*Ceiling* does the following with the input tensor *a*:

\f[
a_{i} = ceiling(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Ceiling">
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
