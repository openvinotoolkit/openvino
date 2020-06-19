## SpaceToDepth <a name="SpaceToDepth"></a>

**Versioned name**: *SpaceToDepth-1*

**Category**: *Data movement*

**Short description**: *SpaceToDepth* operation rearranges data from the spatial dimensions of the input tensor into depth dimension of the output tensor.

**Attributes**

* *block_size*

  * **Description**: *block_size* specifies the size of the value block to be moved. The depth dimension size must be evenly divided by `block_size ^ (len(input.shape) - 2)`.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* *mode*

  * **Description**: specifies how the output depth dimension is gathered from block coordinates and the old depth dimension.
  * **Range of values**:
    * *blocks_first*: the output depth is gathered from `[block_size, ..., block_size,  C]`
    * *depth_first*: the output depth is gathered from `[C, block_size, ..., block_size]`
  * **Type**: `string`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

*   **1**: `data` - input tensor of any type with rank >= 3. Required.

**Outputs**

*   **1**: permuted tensor with shape `[N, C * (block_size ^ K), D1 / block_size, D2 / block_size, ..., DK / block_size]`.

**Detailed description**

*SpaceToDepth* operation permutes element from the input tensor with shape `[N, C, D1, D2, ..., DK]`, to the output tensor where values from the input spatial dimensions `D1, D2, ..., DK` are moved to the new depth dimension. Refer to the [ONNX* specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#SpaceToDepth) for an example of the 4D input tensor case.

The operation is equivalent to the following transformation of the input tensor `data` with `K` spatial dimensions of shape `[N, C, D1, D2, ..., DK]` to *Y* output tensor. If `mode = blocks_first`:

    x' = reshape(data, [N, C, D1/block_size, block_size, D2/block_size, block_size, ... , DK/block_size, block_size])

    x'' = transpose(x',  [0,  3, 5, ..., K + (K + 1), 1,  2, 4, ..., K + K])

    y = reshape(x'', [N, C * (block_size ^ K), D1 / block_size, D2 / block_size, ... , DK / block_size])

If `mode = depth_first`:

    x' = reshape(data, [N, C, D1/block_size, block_size, D2/block_size, block_size, ..., DK/block_size, block_size])

    x'' = transpose(x', [0,  1, 3, 5, ..., K + (K + 1),  2, 4, ..., K + K])

    y = reshape(x'', [N, C * (block_size ^ K), D1 / block_size, D2 / block_size, ..., DK / block_size])


**Example**

```xml
<layer type="SpaceToDepth" ...>
    <data block_size="2" mode="blocks_first"/>
    <input>
        <port id="0">
            <dim>5</dim>
            <dim>7</dim>
            <dim>4</dim>
            <dim>6</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>5</dim>    <!-- data.shape[0] -->
            <dim>28</dim>   <!-- data.shape[1] * (block_size ^ 2) -->
            <dim>2</dim>    <!-- data.shape[2] / block_size -->
            <dim>3</dim>    <!-- data.shape[3] / block_size -->
        </port>
    </output>
</layer>
```