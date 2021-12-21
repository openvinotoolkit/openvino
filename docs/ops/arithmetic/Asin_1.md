# Asin {#openvino_docs_ops_arithmetic_Asin_1}

**Versioned name**: *Asin-1*

**Category**: *Arithmetic unary*

**Short description**: *Asin* performs element-wise inverse sine (arcsin) operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type *T*. **Required.**

**Outputs**

* **1**: The result of element-wise asin operation. A tensor of type *T*.

**Types**

* *T*: any numeric type.

*Asin* does the following with the input tensor *a*:

\f[
a_{i} = asin(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Asin">
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
