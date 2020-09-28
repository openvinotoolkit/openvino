## LessEqual <a name="LessEqual"></a> {#openvino_docs_ops_comparison_LessEqual_1}

**Versioned name**: *LessEqual-1*

**Category**: Comparison binary operation

**Short description**: *LessEqual* performs element-wise comparison operation with two given tensors applying multi-directional broadcast rules.

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

* **1**: A tensor of type T. **Required.**
* **2**: A tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise comparison operation. A tensor of type boolean.

**Types**

* *T*: arbitrary supported type.

**Detailed description**
Before performing arithmetic operation, input tensors *a* and *b* are broadcasted if their shapes are different and `auto_broadcast` attributes is not `none`. Broadcasting is performed according to `auto_broadcast` value.

After broadcasting *LessEqual* does the following with the input tensors *a* and *b*:

\f[
o_{i} = a_{i} <= b_{i}
\f]

**Examples**

*Example 1*

```xml
<layer ... type="LessEqual">
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
<layer ... type="LessEqual">
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