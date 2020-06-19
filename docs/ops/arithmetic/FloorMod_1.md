## FloorMod <a name="FloorMod"></a>

**Versioned name**: *FloorMod-1*

**Category**: Arithmetic binary operation

**Short description**: *FloorMod* returns an element-wise division reminder with two given tensors applying multi-directional broadcast rules. 
The result here is consistent with a flooring divide (like in Python programming language): `floor(x / y) * y + mod(x, y) = x`. 
The sign of the result is equal to a sign of the divisor.

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

* **1**: A tensor of type T. Required.
* **2**: A tensor of type T. Required.

**Outputs**

* **1**: The element-wise division reminder. A tensor of type T.

**Types**

* *T*: any numeric type.

**Examples**

*Example 1*

```xml
<layer ... type="FloorMod">
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
<layer ... type="FloorMod">
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
