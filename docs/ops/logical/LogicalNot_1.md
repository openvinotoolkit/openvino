## LogicalNot <a name="LogicalNot"></a> {#openvino_docs_ops_logical_LogicalNot_1}

**Versioned name**: *LogicalNot-1*

**Category**: Logical unary operation 

**Short description**: *LogicalNot* performs element-wise logical negation operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise logical negation operation. A tensor of type T.

**Types**

* *T*: boolean type.

*LogicalNot* does the following with the input tensor *a*:

\f[
a_{i} = not(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="LogicalNot">
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