## LogicalXor <a name="LogicalXor"></a> {#openvino_docs_ops_logical_LogicalXor_1}

**Versioned name**: *LogicalXor-1*

**Category**: Logical binary operation

**Short description**: *LogicalXor* performs element-wise logical XOR operation with two given tensors applying multi-directional broadcast rules.

**Attributes**:

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input tensors.
  * **Range of values**:
    * *none* - no auto-broadcasting is allowed, all input shapes should match
    * *numpy* - numpy broadcasting rules, aligned with ONNX Broadcasting. Description is available in <a href="https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md">ONNX docs</a>.
  * **Type**: string
  * **Default value**: "numpy"
  * **Required**: *no*

**Inputs**

* **1**: A tensor of type *T*. **Required.**
* **2**: A tensor of type *T*. **Required.**

**Outputs**

* **1**: The result of element-wise logical XOR operation. A tensor of type *T*.

**Types**

* *T*: boolean type.

**Detailed description**
Before performing logical operation, input tensors *a* and *b* are broadcasted if their shapes are different and `auto_broadcast` attributes is not `none`. Broadcasting is performed according to `auto_broadcast` value.

After broadcasting *LogicalXor* does the following with the input tensors *a* and *b*:

\f[
o_{i} = a_{i} xor b_{i}
\f]

**Examples**

*Example 1*

```xml
<layer ... type="LogicalXor">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
        <port id="1">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```

*Example 2: broadcast*
```xml
<layer ... type="LogicalXor">
    <input>
        <port id="0">
            <dim>8</dim>
            <dim>1</dim>
            <dim>6</dim>
            <dim>1</dim>
        </port>
        <port id="1">
            <dim>7</dim>
            <dim>1</dim>
            <dim>5</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>8</dim>
            <dim>7</dim>
            <dim>6</dim>
            <dim>5</dim>
        </port>
    </output>
</layer>
```
