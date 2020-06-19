## Tan <a name="Tan"></a>

**Versioned name**: *Tan-1*

**Category**: Arithmetic unary operation 

**Short description**: *Tan* performs element-wise tangent operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise tan operation. A tensor of type T.

**Types**

* *T*: any numeric type.

*Tan* does the following with the input tensor *a*:

\f[
a_{i} = tan(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Tan">
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

