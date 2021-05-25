## NonZero <a name="NonZero"></a> {#openvino_docs_ops_condition_NonZero_3}

**Versioned name**: *NonZero-3*

**Category**: Condition operation

**Short description**: *NonZero* returns the indices of the non-zero elements of the input tensor. 

**Attributes**

* *output_type*
  * **Description**: the output tensor type
  * **Range of values**:
    * *i64*
    * *i32*
  * **Type**: string
  * **Default value**: "i64"
  * **Required**: *No*

**Inputs**:

*   **1**: `data` tensor of arbitrary rank of type *T*. **Required**.

**Outputs**:

*   **1**: tensor with indices of non-zero elements of shape `[rank(data), num_non_zero]` of type *T_OUT*.

**Types**

* *T*: any numeric type.

* *T_OUT*: Depending on *output_type* attribute can be `int64` or `int32`.

**Detailed description**: *NonZero* returns the indices of the non-zero elements of the input tensor (in row-major order - by dimension).
* The output tensor has shape `[rank(input), num_non_zero]`.
* For example, for the tensor `[[1, 0], [1, 1]]` the output will be `[[0, 1, 1], [0, 0, 1]]`.
* Each output column represents a single non-zero element and for that column, value in row `i` represents this element's index in input's `i`'th dimension.

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
```
Input tensor:   [[0, 1],
                 [1, 0],
                 [1, 0],
                 [1, 1]]

Output: [[0, 1, 2, 3, 3],
         [1, 0, 0, 0, 1]]

Output[0] has indexes of non-zero elements in input for input dimension 0
Output[1] has indexes of non-zero elements in input for input dimension 1
Output[:, i] contains cooridnates the i'th non-zero element
```