## OneHot <a name="OneHot"></a>

**Versioned name**: *OneHot-1*

**Category**: Sequence processing

**Short description**: *OneHot* sets the elements in the output tensor with specified indices to `on_value` and fills all other locations with `off_value`.

**Detailed description**

Taking a tensor with rank `N` as the first input `indices`, OneHot produces tensor with rank `N+1` extending original
tensor with a new dimension at `axis` position in shape. Output tensor is populated with two scalar values: `on_value`
that comes from the 3rd input and `off_value` that comes from the 4nd input. Population is made in the following way:

    output[:, ... ,:, i, :, ... ,:] = on_value if (indices[:, ..., :, :, ..., :] == i) else off_value

where `i` is at `axis` position in `output` shape and has values from range `[0, ..., depth-1]`.

When index element from `indices` is greater or equal to `depth`, it is a well-formed operation. In this case the corresponding row `output[..., i, ...]` is populated with `off_value` only for all `i` values.

Types of input scalars `on_value` and `off_value` should match and can be any of the supported types. The type of output tensor is derived from `on_value` and `off_value`, they all have the same type.

**Attributes**:

* *axis*

  * **Description**: *axis* is a new axis position in the output shape to fill with one-hot values.
  * **Range of values**: an integer. Negative value means counting dimension from the end.
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1**: `indices`: input tensor of rank `N` with indices of any supported integer data type. Can be 0D. Required.
* **2**: `depth`: scalar (0D tensor) of any supported integer type that specifies number of classes and the size of one-hot dimension.
* **3**: `on_value`: scalar (0D tensor) of any type that is the value that the locations in output tensor represented by indices in input take.
* **4**: `off_value`: scalar (0D tensor) of any type that is the value that the locations not represented by indices in input take.

**Outputs**:

* **1** Output tensor of rank `N+1`, where `N` is a rank of input tensor `indices`. A new axis of the size `depth` is inserted at the dimension `axis`.

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