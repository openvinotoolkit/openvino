## ScatterUpdate <a name="ScatterUpdate"></a> {#openvino_docs_ops_movement_ScatterUpdate_3}

**Versioned name**: *ScatterUpdate-3*

**Category**: Data movement operations

**Short description**: *ScatterUpdate* creates a copy of the first input tensor with updated elements specified with second and third input tensors.

**Detailed description**: *ScatterUpdate* creates a copy of the first input tensor with updated elements in positions specified with `indices` input
and values specified with `updates` tensor starting from the dimension with index `axis`. For the `data` tensor of shape `[d_0, d_1, ..., d_n]`,
`indices` tensor of shape `[i_0, i_1, ..., i_k]` and `updates` tensor of shape
`[d_0, d_1, ... d_(axis - 1), i_0, i_1, ..., i_k, d_(axis + 1), ..., d_n]` the operation computes
for each `m, n, ..., p` of the `indices` tensor indices:

```
data[..., indices[m, n, ..., p], ...] = updates[..., m, n, ..., p, ...]
```

where first `...` in the `data` corresponds to first `axis` dimensions, last `...` in the `data` corresponds to the
`rank(data) - (axis + 1)` dimensions.

Several examples for case when `axis = 0`:
1. `indices` is a 0D tensor: `data[indices, ...] = updates[...]`
2. `indices` is a 1D tensor (for each `i`): `data[indices[i], ...] = updates[i, ...]`
3. `indices` is a ND tensor (for each `i, ..., j`): `data[indices[i, ..., j], ...] = updates[i, ..., j, ...]`

This operation is similar to TensorFlow* operation [ScatterUpdate](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/scatter_update)
but allows scattering for the arbitrary axis.

**Attributes**: *ScatterUpdate* does not have attributes.

**Inputs**:

*   **1**: `data` tensor of arbitrary rank `r` and of type *T*. **Required.**

*   **2**: `indices` tensor with indices of type *T_IND*.
All index values are expected to be within bounds `[0, s - 1]` along axis of size `s`. If multiple indices point to the
same output location then the order of updating the values is undefined. If an index points to non-existing output
tensor element or is negative then an exception is raised. **Required.**

*   **3**: `updates` tensor of type *T*. **Required.**

*   **4**: `axis` tensor with scalar or 1D tensor with one element of type *T_AXIS* specifying axis for scatter.
The value can be in range `[-r, r - 1]` where `r` is the rank of `data`. **Required.**

**Outputs**:

*   **1**: tensor with shape equal to `data` tensor of the type *T*.

**Types**

* *T*: any numeric type.

* *T_IND*: any supported integer types.

* *T_AXIS*: any supported integer types.

**Example**

```xml
<layer ... type="ScatterUpdate">
    <input>
        <port id="0">
            <dim>1000</dim>
            <dim>256</dim>
            <dim>10</dim>
            <dim>15</dim>
        </port>
        <port id="1">
            <dim>125</dim>
            <dim>20</dim>
        </port>
        <port id="2">
            <dim>1000</dim>
            <dim>125</dim>
            <dim>20</dim>
            <dim>10</dim>
            <dim>15</dim>
        </port>
        <port id="3">     <!-- value [1] -->
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="4" precision="FP32">
            <dim>1000</dim>
            <dim>256</dim>
            <dim>10</dim>
            <dim>15</dim>
        </port>
    </output>
</layer>
```
