## ShuffleChannels <a name="ShuffleChannels"></a>

**Versioned name**: *ShuffleChannels-1*

**Name**: *ShuffleChannels*

**Category**: Data movement

**Short description**: *ShuffleChannels* permutes data in the channel dimension of the input tensor.

**Attributes**:

* *axis*

  * **Description**: *axis* specifies the index of a channel dimension.
  * **Range of values**: an integer number in the range [-4, 3]
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *No*

* *group*

  * **Description**: *group* specifies the number of groups to split the channel dimension into. This number must evenly divide the channel dimension size.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *No*

**Inputs**:

*   **1**: 4D input tensor of any supported data type. Required.

**Outputs**:

*   **1**: 4D input tensor with shape and element type as for the input tensor.

**Mathematical Formulation**

The operation is the equivalent with the following transformation of the input tensor *x* of shape *[N, C, H, W]*:

```
x' = reshape(x, [N, group, C / group, H * W])
x'' = transpose(x', [0, 2, 1, 3])
y = reshape(x'', [N, C, H, W])
```

where `group` is the layer parameter described above and the `axis = 1`.

**Example**

```xml
<layer ... type="ShuffleChannels" ...>
    <data group="3" axis="1"/>
    <input>
        <port id="0">
            <dim>5</dim>
            <dim>12</dim>
            <dim>200</dim>
            <dim>400</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>5</dim>
            <dim>12</dim>
            <dim>200</dim>
            <dim>400</dim>
        </port>
    </output>
</layer>
```