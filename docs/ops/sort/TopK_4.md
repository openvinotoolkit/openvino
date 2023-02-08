`# TopK  {#openvino_docs_ops_sort_TopK_4}

**Versioned name**: *TopK-4*

**Category**: *Sorting and maximization*

**Short description**: *TopK* computes indices and values of the *k* maximum/minimum values for each slice along a specified axis.

**Attributes**

* *axis*

  * **Description**: Specifies the axis along which the values are retrieved.
  * **Range of values**: An integer. Negative values means counting dimension from the back.
  * **Type**: `int`
  * **Required**: *yes*

* *mode*

  * **Description**: Specifies whether TopK selects the largest or the smallest elements from each slice.
  * **Range of values**: "min", "max"
  * **Type**: `string`
  * **Required**: *yes*

* *sort*

  * **Description**: Specifies the order of corresponding elements of the output tensor.
  * **Range of values**: `value`, `index`, `none`
  * **Type**: `string`
  * **Required**: *yes*

* *stable*

  * **Description**: Specifies whether the equivalent elements should maintain their relative order from the input tensor. Takes effect only if sort is set to `value` or `index`.
  * **Range of values**: `true` of `false`
  * **Type**: `boolean`
  * **Default value**: `false`
  * **Required**: *no*

* *index_element_type*

  * **Description**: the type of output tensor with indices
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i32"
  * **Required**: *no*


**Inputs**:

*   **1**: tensor with arbitrary rank and type *T*. **Required.**

*   **2**: *k* -- a scalar of any integer type that specifies how many elements from the input tensor should be selected. **Required.**

**Outputs**:

*   **1**: Output tensor of type *T* with at most *k* values from the input tensor along specified dimension *axis*. The shape of the tensor is `[input1.shape[0], ..., input1.shape[axis-1], 1..k, input1.shape[axis+1], ..., input1.shape[input1.rank - 1]]`.

*   **2**: Output tensor containing indices of the corresponding elements(values) from the first output tensor. The indices point to the location of selected values in the original input tensor. The shape of this output tensor is the same as the shape of the 1st output, that is `[input1.shape[0], ..., input1.shape[axis-1], 1..k, input1.shape[axis+1], ..., input1.shape[input1.rank - 1]]`. The type of this tensor *T_IND* is controlled by the `index_element_type` attribute.

**Types**

* *T*: any numeric type.

* *T_IND*: `int64` or `int32`.

**Detailed Description**

The output tensor is populated by values computed in the following way:

    output[i1, ..., i(axis-1), j, i(axis+1) ..., iN] = top_k(input[i1, ...., i(axis-1), :, i(axis+1), ..., iN]), k, sort, mode)

So for each slice `input[i1, ...., i(axis-1), :, i(axis+1), ..., iN]` *TopK* values are computed individually.

Sorting and minimum/maximum are controlled by `sort` and `mode` attributes:
  * *mode*=`max`, *sort*=`value` - descending by value
  * *mode*=`max`, *sort*=`index` - ascending by index
  * *mode*=`max`, *sort*=`none`  - undefined
  * *mode*=`min`, *sort*=`value` - ascending by value
  * *mode*=`min`, *sort*=`index` - ascending by index
  * *mode*=`min`, *sort*=`none`  - undefined

The relative order of equivalent elements in a given slice is only preserved if the *stable* attribute is set to `true`. This makes the implementation use stable sorting algorithm during the computation of TopK elements. Otherwise the output order is undefined.

**Example**

This example assumes that `K` is equal to 10:

```xml
<layer ... type="TopK" ... >
    <data axis="3" mode="max" sort="value" stable="true" index_element_type="i64"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
        <port id="1">
        </port>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>3</dim>
            <dim>10</dim>
            <dim>224</dim>
        </port>
        <port id="3">
            <dim>1</dim>
            <dim>3</dim>
            <dim>10</dim>
            <dim>224</dim>
        </port>
    </output>
</layer>
```
