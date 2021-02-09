## OneHot <a name="OneHot"></a> {#openvino_docs_ops_sequence_OneHot_1}

**Versioned name**: *OneHot-1*

**Category**: Sequence processing

**Short description**: *OneHot* sets the elements in the output tensor with specified indices to `on_value` and fills all other locations with `off_value`.

**Detailed description**

Taking a tensor with rank `N` as the first input `indices`, OneHot produces a tensor with rank `N+1` extending the original
tensor with a new dimension at the `axis` position. The output tensor is populated with two scalar values: `on_value`
that comes from the 3rd input and `off_value` that comes from the 4nd input. The population is made in the following way:

    output[:, ... ,:, i, :, ... ,:] = on_value if (indices[:, ..., :, :, ..., :] == i) else off_value

where `i` is at the `axis` position in the `output` shape and has values from the range `[0, ..., depth-1]`.

When some elements from the `indices` are greater or equal to the `depth`, it is a well-formed operation. The corresponding output rows are populated with `off_value` in this case.

The types of input scalars `on_value` and `off_value` should match and be equal to any supported type. The output tensor type is derived from the `on_value` or the `off_value`, they all have the same type.

**Attributes**:

* *axis*

  * **Description**: *axis* is a new axis position in the output shape to fill with one-hot values.
  * **Range of values**: an integer. Negative value means counting dimension from the end.
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1**: `indices`: input tensor with non-negative indices of any supported integer data type. Can be 0D. Required.
* **2**: `depth`: positive scalar (0D tensor) of any supported integer type that specifies the number of classes and thus the size of the one-hot dimension. Required.
* **3**: `on_value`: scalar (0D tensor) of any supported type that fills the locations in output tensor specified in `indices`. Required.
* **4**: `off_value`: scalar (0D tensor) of the same type as `on_value` that fills the locations not represented in `indices`. Required.

**Outputs**:

* **1** A tensor of rank `N+1`, where `N` is a rank of the input tensor `indices`. A new axis of the size `depth` is inserted at the dimension `axis`. The output type is the same as the `on_value` type.

**Examples**

```xml
<layer ... type="OneHot" ...>
    <data axis="-1"/>
    <input>
        <port id="0">    <!-- indices value: [0, 1, 2] -->
            <dim>3</dim>
        </port>
        <port id="1">    <!-- depth value: 2 -->
        </port>
        <port id="2">    <!-- on_value 5 -->
        </port>
        <port id="3">    <!-- off_value 10 -->
        </port>
    </input>
    <output>
        <port id="0">    <!-- output value # [[5, 10], [10, 5], [10, 10]] -->
            <dim>3</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```