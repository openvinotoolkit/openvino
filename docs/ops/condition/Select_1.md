## Select <a name="Select"></a>

**Versioned name**: *Select-1*

**Category**: *Conditions*

**Short description**: *Select* returns a tensor filled with the elements from the second or the third inputs, depending on the condition (the first input) value.

**Detailed description**

*Select* takes elements from `then` input tensor or the `else` input tensor based on a condition mask
 provided in the first input `cond`. Before performing selection, input tensors `then` and `else` are broadcasted to each other if their shapes are different and `auto_broadcast` attributes is not `none`. Then the `cond` tensor is one-way broadcasted to the resulting shape of broadcasted `then` and `else`. Broadcasting is performed according to `auto_broadcast` value.

**Attributes**

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input tensors.
  * **Range of values**:
    * *none* - no auto-broadcasting is allowed, all input shapes should match
    * *numpy* - numpy broadcasting rules, aligned with ONNX Broadcasting. Description is available in <a href="https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md">ONNX docs</a>.
  * **Type**: string
  * **Default value**: "numpy"
  * **Required**: *no*


**Inputs**:

* **1**: `cond` tensor with selection mask of type `boolean`. The tensor can be 0D.

* **2**: `then` the tensor with elements to take where the corresponding element in `cond` is true. Arbitrary type that should match type of `else` input tensor.

* **3**: `else` the tensor with elements to take where the corresponding element in `cond` is false. Arbitrary type that should match type of `then` input tensor.


**Outputs**:

* **1**: blended output tensor that is tailored from values of inputs tensors `then` and `else` based on `cond` and broadcasting rules. It has the same type of elements as `then` and `else`.


**Example**

```xml
<layer ... type="Select">
    <input>
        <port id="0">     <!-- cond value is: [[false, false], [true, false], [true, true]] -->
            <dim>3</dim>
            <dim>2</dim>
        </port>
        <port id="1">     <!-- then value is: [[-1, 0], [1, 2], [3, 4]] -->
            <dim>3</dim>
            <dim>2</dim>
        </port>
        <port id="2">     <!-- else value is: [[11, 10], [9, 8], [7, 6]] -->
            <dim>3</dim>
            <dim>2</dim>
        </port>
    </input>
    <output>
        <port id="1">     <!-- output value is: [[11, 10], [1, 8], [3, 4]] -->
            <dim>3</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```