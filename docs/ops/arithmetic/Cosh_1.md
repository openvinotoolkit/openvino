## Cosh <a name="Cosh"></a>

**Versioned name**: *Cosh-1*

**Category**: Arithmetic unary operation 

**Short description**: *Cosh* performs element-wise hyperbolic cosine operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise cosh operation. A tensor of type T.

**Types**

* *T*: any numeric type.

*Cosh* does the following with the input tensor *a*:

\f[
a_{i} = cosh(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Cosh">
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
