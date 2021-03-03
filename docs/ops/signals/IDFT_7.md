## IDFT <a name="IDFT"></a> {#openvino_docs_ops_signals_IDFT_7}

**Versioned name**: *IDFT-7*

**Category**: Image processing

**Short description**: *IDFT* layer performs the discrete inverse fast Fourier transformation of input tensor by specified dimensions.

**Detailed description**: *IDFT* performs the discrete inverse fast Fourier transformation of input tensor with respect to specified axes.

**Attributes**:

    No attributes available.

**Inputs**

*   **1**: `data` - Input tensor of type *T* with data for the IDFT transformation. Type of elements is any supported floating point type or `int8` type. Required.
*   **2**: `axes` - 1D tensor of type *T_IND* specifying dimension indices where IDFT is applied, and `axes` is any unordered list of indices of different dimensions of input tensor, e.g. `[0, 4]`, `[4, 0]`, `[4, 2, 1]`, `[1, 2, 3]`. These indices should be non-negative integers from `0` to `rank(data) - 1` inclusively.  Other dimensions do not change. The order of elements in `axes` attribute matters, and mapped directly to elements in the 3d input `signal_size`. Required.
*   **3**: `signal_size` - 1D tensor of type *T_IND* describing signal size with respect to axes from the input `axes`. For any `i in range(0, len(axes))`, if `signal_size[i] == -1`, then IDFT is calculated for full size of the axis `axes[i]`. If `signal_size[i] > input_shape[axes[i]]`, then input data are zero-padded with respect to the axis `axes[i]`. Finally, `signal_size[i] < input_shape[axes[i]]`, then input data are trimmedwith respect to the axis `axes[i]`. Optional, with default value `[input_shape[a] for a in axes]`.
*   **Note**: The following constraint must be satisfied: `rank(data) >= len(axes) + 1 and input_shape[-1] == 2 and (rank(data) - 1) not in axes`.

**Outputs**

*   **1**: Resulting tensor with elements of the same type as input `data` tensor. The shape of the output matches input `data` shape except dimensions mentioned in `axes` input. For other dimensions shape matches sizes from `signal_size` in order specified in `axes`.

**Types**

* *T*: floating point type.

* *T_IND*: `int64` or `int32`.

**Example**:

There is no `signal_size` input:
```xml
<layer ... type="IDFT" ... >
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>320</dim>
            <dim>320</dim>
            <dim>2</dim>
        </port>
        <port id="1">
            <dim>2</dim> <!-- [1, 2] -->
        </port>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>320</dim>
            <dim>320</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```


There is `signal_size` input:
```xml
<layer ... type="IDFT" ... >
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>320</dim>
            <dim>320</dim>
            <dim>2</dim>
        </port>
        <port id="1">
            <dim>2</dim> <!-- [1, 2] -->
        </port>
        <port id="2">
            <dim>2</dim> <!-- [512, 100] -->
        </port>
    <output>
        <port id="3">
            <dim>1</dim>
            <dim>512</dim>
            <dim>100</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```