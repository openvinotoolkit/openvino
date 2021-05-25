## ShuffleChannels <a name="ShuffleChannels"></a> {#openvino_docs_ops_movement_ShuffleChannels_1}

**Versioned name**: *ShuffleChannels-1*

**Name**: *ShuffleChannels*

**Category**: Data movement

**Short description**: *ShuffleChannels* permutes data in the channel dimension of the input tensor.

**Detailed description**:

Input ND tensor of data shape is always interpreted as 4D tensor with the following shape:

    dim 0: data_shape[0] * data_shape[1] * ... * data_shape[axis-1] (or 1 if axis == 0)
    dim 1: group
    dim 2: data_shape[axis] / group
    dim 3: data_shape[axis+1] * data_shape[axis+2] * ... * data_shape[data_shape.size()-1] (or 1 if axis points to last dimension)

So the dimensions before or after the dimention pointed by `axis` are flattened and reshaped back to the original shape after channnels shuffling.


The operation is the equivalent with the following transformation of the input tensor *x* of shape *[N, C, H, W]*:

\f[
\begin{*align}
x' = reshape(x, [N, group, C / group, H * W])\\
x'' = transpose(x', [0, 2, 1, 3])\\
y = reshape(x'', [N, C, H, W])\\
\end{*align}
\f]

where `group` is the layer attribute described below and the `axis = 1`.

**Attributes**:

* *axis*

  * **Description**: *axis* specifies the index of a channel dimension.
  * **Range of values**: an integer number in the range [-input_rank, input_rank-1]
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

*   **1**: ND (1D or bigger) input tensor of type *T*. **Required.**

**Outputs**:

*   **1**: Output tensor with element type *T* and same shape as the input tensor.

**Types**

* *T*: any supported numeric type.

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
