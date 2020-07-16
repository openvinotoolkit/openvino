## Log <a name="Log"></a> {#openvino_docs_ops_arithmetic_Log_1}

**Versioned name**: *Log-1*

**Category**: Arithmetic unary operation 

**Short description**: *Log* performs element-wise natural logarithm operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise log operation. A tensor of type T.

**Types**

* *T*: any numeric type.

*Log* does the following with the input tensor *a*:

\f[
a_{i} = log(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Log">
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