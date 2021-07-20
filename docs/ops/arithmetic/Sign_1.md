## Sign <a name="Sign"></a> {#openvino_docs_ops_arithmetic_Sign_1}

**Versioned name**: *Sign-1*

**Category**: Arithmetic unary operation

**Short description**: *Sign* performs element-wise sign operation on a given input tensor.

**Detailed description**: *Sign* performs element-wise sign operation on a given input tensor, based on the following mathematical formula:

\f[
a_{i} = sign(a_{i})
\f]

**Attributes**: *Sign* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise *Sign* operation. A tensor of type *T* with mapped elements of the input tensor to -1 (if it is negative), 0 (if it is zero), or 1 (if it is positive).

**Types**

* *T*: any numeric type.


**Example**

```xml
<layer ... type="Sign">
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
