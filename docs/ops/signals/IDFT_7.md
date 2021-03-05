## IDFT <a name="IDFT"></a> {#openvino_docs_ops_signals_IDFT_7}

**Versioned name**: *IDFT-7*

**Category**: Signal processing

**Short description**: *IDFT* layer performs the inverse discrete Fourier transformation of input tensor by specified dimensions.

**Detailed description**: *IDFT* performs the inverse discrete Fourier transformation of input tensor with respect to specified axes.

**Attributes**:

    No attributes available.

**Inputs**

*   **1**: `data` - Input tensor of type *T* with data for the IDFT transformation. Type of elements is any supported floating point type. The last dimension of the input tensor must be equal to 2, i.e. the input tensor shape must have the form `[D_0, D_1, ..., D_{N-1}, 2]`, representing the real and imaginary components of complex numbers in `[:, ..., :, 0]` and in `[:, ..., :, 1]` correspondingly.  Required.
*   **2**: `axes` - 1D tensor of type *T_IND* specifying dimension indices where IDFT is applied, and `axes` is any unordered list of indices of different dimensions of input tensor, e.g. `[0, 4]`, `[4, 0]`, `[4, 2, 1]`, `[1, 2, 3]`. These indices should be non-negative integers from `0` to `rank(data) - 1` inclusively.  Other dimensions do not change. The order of elements in `axes` attribute matters, and mapped directly to elements in the 3d input `signal_size`. Required.
*   **Note**: The following constraint must be satisfied: `rank(data) >= len(axes) + 1 and input_shape[-1] == 2 and (rank(data) - 1) not in axes`.
*   **3**: `signal_size` - 1D tensor of type *T_IND* describing signal size with respect to axes from the input `axes`. If `signal_size[i] == -1`, then IDFT is calculated for full size of the axis `axes[i]`. If `signal_size[i] > input_shape[axes[i]]`, then input data are zero-padded with respect to the axis `axes[i]`. Finally, `signal_size[i] < input_shape[axes[i]]`, then input data are trimmed with respect to the axis `axes[i]`. More precisely, if `signal_size[i] < input_shape[axes[i]]`, the slice `0: signal_size[i]` of the axis `axes[i]` is considered. This behaviour is [the same as in PyTorch](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SpectralOps.cpp#L88). TensorFlow always uses full sizes of transformed axes. Optional, with default value `[input_shape[a] for a in axes]`.
*   **Note**: If the input `signal_size` is specified, then the size of `signal_size` must be the same as the size of `axes`.

**Outputs**

*   **1**: Resulting tensor with elements of the same type as input `data` tensor. The shape of the output is calculated as the following. If the input `signal_size` is not specified, then the shape of output is the same as the shape of `data`. Otherwise, `output_shape[axis] = input_shape[axis]` for `axis not in axes`, and if `signal_size[i] == -1` then `output_shape[axes[i]] = input_shape[axes[i]]`, else `output_shape[axes[i]] = signal_size[i]`.

**Types**

* *T*: floating point type.

* *T_IND*: `int64` or `int32`.

**Detailed description**: *IDFT* performs the discrete Fourier transformation of input tensor with respect to specified axes. Calculations are performed according to the following rules.

For simplicity, assume that an input tensor `A` has the shape `[B_0, ..., B_{k-1}, M_0, ..., M_{r-1}, 2]`, `axes=[k+1,...,k+r]`, and `signal_size=[S_0,...,S_{r-1}]`.

Let `D` be an input tensor `A`, taking into account the `signal_size`, and, hence, `D` has the shape `[B_0, ..., B_{k-1}, S_0, ..., S_{r-1}, 2]`.

Next, put
\f[X[j_0,\dots,j_{k-1},j_k,\dots,j_{k+r}]=D[j_0,\dots,j_{k-1},j_k,\dots,j_{k+r},0]+iD[j_0,\dots,j_{k-1},j_k,\dots,j_{k+r},1]\f]
for all indices `j_0,...,j_{k+r}`, where `i` is an imaginary unit, i.e. `X` is a complex tensor.

Then the discrete Fourier transform is the tensor `Y` (the shapes of the tensors `X` and `Y` are the same) such that
\f[Y[n_0,\dots,n_{k-1},m_0,\dots,m_{r-1}]=\frac{1}{\prod\limits_{j=0}^{r-1}S_j}\sum\limits_{p_0=0}^{S_0}\cdots\sum\limits_{p_{r-1}=0}^{S_{r-1}}X[n_0,\dots,n_{k-1},j_0,\dots,j_{r-1}]\exp\left(2\pi i\sum\limits_{q=0}^{r-1}\frac{m_qj_q}{S_s}\right)\f]
for all indices `n_0,...,n_{k-1}`, `m_0,...,m_{r-1}`, and the result of the operation is the real tensor `Z` with the shape `[B_0, ..., B_{k-1}, S_0, ..., S_{r-1}, 2]` and such that
\f[Z[n_0,\dots,n_{k-1},m_0,\dots,m_{r-1}, 0]=Re Y[n_0,\dots,n_{k-1},m_0,\dots,m_{r-1}],\f]
\f[Z[n_0,\dots,n_{k-1},m_0,\dots,m_{r-1}, 1]=Im Y[n_0,\dots,n_{k-1},m_0,\dots,m_{r-1}].\f]

Calculations for the generic case of axes and signal sizes are similar.

**Example**:

There is no `signal_size` input (4D input tensor):
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

There is no `signal_size` input (3D input tensor):
```xml
<layer ... type="IDFT" ... >
    <input>
        <port id="0">
            <dim>320</dim>
            <dim>320</dim>
            <dim>2</dim>
        </port>
        <port id="1">
            <dim>2</dim> <!-- [0, 1] -->
        </port>
    <output>
        <port id="2">
            <dim>320</dim>
            <dim>320</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```


There is `signal_size` input (4D input tensor):
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


There is `signal_size` input (3D input tensor):
```xml
<layer ... type="IDFT" ... >
    <input>
        <port id="0">
            <dim>320</dim>
            <dim>320</dim>
            <dim>2</dim>
        </port>
        <port id="1">
            <dim>2</dim> <!-- [0, 1] -->
        </port>
        <port id="2">
            <dim>2</dim> <!-- [512, 100] -->
        </port>
    <output>
        <port id="3">
            <dim>512</dim>
            <dim>100</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```


There is `signal_size` input (5D input tensor, `-1` in `signal_size`, unsorted axes):
```xml
<layer ... type="IDFT" ... >
    <input>
        <port id="0">
            <dim>16</dim>
            <dim>768</dim>
            <dim>580</dim>
            <dim>320</dim>
            <dim>2</dim>
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
            <dim>768</dim>
            <dim>1024</dim>
            <dim>170</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```


There is `signal_size` input (5D input tensor, `-1` in `signal_size`, unsorted axes, the second example):
```xml
<layer ... type="IDFT" ... >
    <input>
        <port id="0">
            <dim>16</dim>
            <dim>768</dim>
            <dim>580</dim>
            <dim>320</dim>
            <dim>2</dim>
        </port>
        <port id="1">
            <dim>3</dim> <!-- axes input contains  [3, 0, 2] -->
        </port>
        <port id="2">
            <dim>3</dim> <!-- signal_size input contains [258, -1, 2056] -->
        </port>
    <output>
        <port id="3">
            <dim>16</dim>
            <dim>768</dim>
            <dim>2056</dim>
            <dim>258</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```
