# Discrete Fourier Transformation for real-valued input (RFFT) {#openvino_docs_ops_signals_RFFT_9}

**Versioned name**: *RFFT-9*

**Category**: *Signal processing*

**Short description**: *RFFT* operation performs the discrete real-to-complex Fourier transformation of input tensor by specified dimensions.

**Attributes**:

    No attributes available.

**Inputs**

*   **1**: `data` - Input tensor of type *T* with data for the RFFT transformation. Type of elements is any supported floating-point type. **Required.**
*   **2**: `axes` - 1D tensor of type *T_IND* specifying dimension indices where RFFT is applied, and `axes` is any unordered list of indices of different dimensions of input tensor, for example, `[0, 4]`, `[4, 0]`, `[4, 2, 1]`, `[1, 2, 3]`, `[-3, 0, -2]`. These indices should be integers from `-r` to `r - 1` inclusively, where `r = rank(data)`. A negative axis `a` is interpreted as an axis `r + a`. Other dimensions do not change. The order of elements in `axes` attribute matters, and is mapped directly to elements in the third input `signal_size`. **Required.**
*   **3**: `signal_size` - 1D tensor of type *T_SIZE* describing signal size with respect to axes from the input `axes`. If `signal_size[i] == -1`, then RFFT is calculated for full size of the axis `axes[i]`. If `signal_size[i] > input_shape[axes[i]]`, then input data are zero-padded with respect to the axis `axes[i]` at the end. Finally, `signal_size[i] < input_shape[axes[i]]`, then input data are trimmed with respect to the axis `axes[i]`. More precisely, if `signal_size[i] < input_shape[axes[i]]`, the slice `0: signal_size[i]` of the axis `axes[i]` is considered. Optional, with default value `[input_shape[a] for a in axes]`.
*   **NOTE**: If the input `signal_size` is specified, the size of `signal_size` must be the same as the size of `axes`.

**Outputs**

*   **1**: Resulting tensor with elements of the same type as input `data` tensor and with rank `r + 1`, where `r = rank(data)`. The shape of the output has the form `[S_0, S_1, ..., S_{r-1}, 2]`, where all `S_a` are calculated as follows. Firstly, we calculate `normalized_axes`, where each `normalized_axes[i] = axes[i]`, if `axes[i] >= 0`, and `normalized_axes[i] = axes[i] + r` otherwise. Next, if `a not in normalized_axes`, then `S_a = input_shape[a]`. Suppose that `a in normalized_axes`, that is `a = normalized_axes[i]` for some `i`. In such case, `S_a = input_shape[a] // 2 + 1` if the `signal_size` input is not specified, or, if it is specified, `signal_size[i] = -1`; and `S_a = signal_size[a] // 2 + 1` otherwise.

**Types**

* *T*: floating-point type.

* *T_IND*: `int64` or `int32`.

* *T_SIZE*: `int64` or `int32`.

**Detailed description**: *RFFT* performs the discrete Fourier transformation of real-valued input tensor with respect to specified axes. Calculations are performed according to the following rules.

For simplicity, assume that an input tensor `A` has the shape `[B_0, ..., B_{k-1}, M_0, ..., M_{r-1}]`, `axes=[k+1,...,k+r]`, and `signal_size=[S_0,...,S_{r-1}]`.

Let `D` be an input tensor `A`, taking into account the `signal_size`, and, hence, `D` has the shape `[B_0, ..., B_{k-1}, S_0, ..., S_{r-1}]`.

Next, let
\f[X=X[j_0,\dots,j_{k-1},j_k,\dots,j_{k+r}]\f]
for all indices `j_0,...,j_{k+r}`, be a real-valued input tensor.

Then the transformation RFFT of the tensor `X` is the tensor `Y` of the shape `[B_0, ..., B_{k-1}, S_0 // 2 + 1, ..., S_{r-1} // 2 + 1]`, such that
\f[Y[n_0,\dots,n_{k-1},m_0,\dots,m_{r-1}]=\sum\limits_{p_0=0}^{S_0}\cdots\sum\limits_{p_{r-1}=0}^{S_{r-1}}X[n_0,\dots,n_{k-1},j_0,\dots,j_{r-1}]\exp\left(-2\pi i\sum\limits_{q=0}^{r-1}\frac{m_qj_q}{S_s}\right)\f]
for all indices `n_0,...,n_{k-1}`, `m_0,...,m_{r-1}`.

Calculations for the generic case of axes and signal sizes are similar.

**Example**:

There is no `signal_size` input (3D input tensor):
```xml
<layer ... type="RFFT" ... >
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>320</dim>
            <dim>320</dim>
        </port>
        <port id="1">
            <dim>2</dim> <!-- axes input contains [1, 2] -->
        </port>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>161</dim>
            <dim>161</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```

There is no `signal_size` input (2D input tensor):
```xml
<layer ... type="RFFT" ... >
    <input>
        <port id="0">
            <dim>320</dim>
            <dim>320</dim>
        </port>
        <port id="1">
            <dim>2</dim> <!-- axes input contains [0, 1] -->
        </port>
    <output>
        <port id="2">
            <dim>161</dim>
            <dim>161</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```


There is `signal_size` input (3D input tensor):
```xml
<layer ... type="RFFT" ... >
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>320</dim>
            <dim>320</dim>
        </port>
        <port id="1">
            <dim>2</dim> <!-- axes input contains [1, 2] -->
        </port>
        <port id="2">
            <dim>2</dim> <!-- signal_size input contains [512, 100] -->
        </port>
    <output>
        <port id="3">
            <dim>1</dim>
            <dim>257</dim>
            <dim>51</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```


There is `signal_size` input (2D input tensor):
```xml
<layer ... type="RFFT" ... >
    <input>
        <port id="0">
            <dim>320</dim>
            <dim>320</dim>
        </port>
        <port id="1">
            <dim>2</dim> <!-- axes input contains [0, 1] -->
        </port>
        <port id="2">
            <dim>2</dim> <!-- signal_size input contains [512, 100] -->
        </port>
    <output>
        <port id="3">
            <dim>257</dim>
            <dim>51</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```


There is `signal_size` input (4D input tensor, `-1` in `signal_size`, unsorted axes):
```xml
<layer ... type="RFFT" ... >
    <input>
        <port id="0">
            <dim>16</dim>
            <dim>768</dim>
            <dim>580</dim>
            <dim>320</dim>
        </port>
        <port id="1">
            <dim>3</dim> <!-- axes input contains  [3, 1, 2] -->
        </port>
        <port id="2">
            <dim>3</dim> <!-- signal_size input contains [170, -1, 1024] -->
        </port>
    <output>
        <port id="3">
            <dim>16</dim>
            <dim>385</dim>
            <dim>513</dim>
            <dim>86</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```


There is `signal_size` input (4D input tensor, `-1` in `signal_size`, unsorted axes, the second example):
```xml
<layer ... type="RFFT" ... >
    <input>
        <port id="0">
            <dim>16</dim>
            <dim>768</dim>
            <dim>580</dim>
            <dim>320</dim>
        </port>
        <port id="1">
            <dim>3</dim> <!-- axes input contains  [3, 0, 2] -->
        </port>
        <port id="2">
            <dim>3</dim> <!-- signal_size input contains [258, -1, 2056] -->
        </port>
    <output>
        <port id="3">
            <dim>9</dim>
            <dim>768</dim>
            <dim>1029</dim>
            <dim>130</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```
