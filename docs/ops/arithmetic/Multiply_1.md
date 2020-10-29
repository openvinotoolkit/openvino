## Multiply <a name="Multiply"></a> {#openvino_docs_ops_arithmetic_Multiply_1}

**Versioned name**: *Multiply-1*

**Category**: Arithmetic binary operation

**Short description**: *Multiply* performs element-wise multiplication operation with two given tensors applying multi-directional broadcast rules.

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

* **1**: The result of element-wise multiplication operation. A tensor of type T.

**Types**

* *T*: any numeric type.

**Detailed description**
Before performing arithmetic operation, input tensors *a* and *b* are broadcasted if their shapes are different and `auto_broadcast` attributes is not `none`. Broadcasting is performed according to `auto_broadcast` value.

After broadcasting *Multiply* does the following with the input tensors *a* and *b*:

\f[
o_{i} = a_{i} * b_{i}
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Multiply">
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
<layer ... type="Multiply">
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