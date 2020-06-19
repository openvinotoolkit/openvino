## Cos <a name="Cos"></a>

**Versioned name**: *Cos-1*

**Category**: Arithmetic unary operation 

**Short description**: *Cos* performs element-wise cosine operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise cos operation. A tensor of type T.

**Types**

* *T*: any numeric type.

*Cos* does the following with the input tensor *a*:

\f[
a_{i} = cos(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Cos">
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
