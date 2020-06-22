## NonZero <a name="NonZero"></a>

**Versioned name**: *NonZero-3*

**Category**: Condition operation

**Short description**: *NonZero* returns the indices of the non-zero elements of the input tensor.

**Detailed description**: *NonZero* returns the indices of the non-zero elements of the input tensor (in row-major order - by dimension).
The output tensor has shape `[rank(input), num_non_zero]`. For example, for the tensor `[[1, 0], [1, 1]]` the output will be `[[0, 1, 1], [0, 0, 1]]`.

**Attributes**

* *output_type*

  * **Description**: the output tensor type
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i64"
  * **Required**: *No*

**Inputs**:

*   **1**: `data` tensor of arbitrary rank of type *T*. Required.

**Outputs**:

*   **1**: tensor with indices of non-zero elements of shape `[rank(data), num_non_zero]` of type *T_IND*.

**Types**

* *T*: any numeric type.

* *T_IND*: `int64` or `int32`.

**Example**

```xml
<layer ... type="NonZero">
    <data output_type="i64"/>
    <input>
        <port id="0">
            <dim>3</dim>
            <dim>10</dim>
            <dim>100</dim>
            <dim>200</dim>
        </port>
     </input>
    <output>
        <port id="1">
            <dim>4</dim>
            <dim>600000</dim>
        </port>
    </output>
</layer>
```
