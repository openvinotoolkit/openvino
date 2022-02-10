# Inverse Discrete complex-to-real Fourier Transformation (IRFFT) {#openvino_docs_ops_signals_IRFFT_9}

**Versioned name**: *IRFFT-9*

**Category**: *Signal processing*

**Short description**: *IRFFT* operation performs the inverse complex-to-real discrete Fourier transformation of input tensor by specified dimensions.

**Attributes**:

    No attributes available.

**Inputs**

*   **1**: `data` - Input tensor of type *T* with data for the IRFFT transformation. Type of elements is any supported floating-point type. The last dimension of the input tensor must be equal to 2, that is the input tensor shape must have the form `[D_0, D_1, ..., D_{N-1}, 2]`, representing the real and imaginary components of complex numbers in `[:, ..., :, 0]` and in `[:, ..., :, 1]` correspondingly. **Required.**
*   **2**: **2**: `axes` - 1D tensor of type *T_IND* specifying dimension indices where IRFFT is applied, and `axes` is any unordered list of indices of different dimensions of input tensor, for example, `[0, 4]`, `[4, 0]`, `[4, 2, 1]`, `[1, 2, 3]`, `[-3, 0, -2]`. These indices should be integers from `-(r - 1)` to `(r - 2)` inclusively, where `r = rank(data)`. A negative axis `a` is interpreted as an axis `r - 1 + a`. Other dimensions do not change. The order of elements in `axes` attribute matters, and is mapped directly to elements in the third input `signal_size`. **Required.**
*   **NOTE**: The following constraint must be satisfied: `rank(data) >= len(axes) + 1 and input_shape[-1] == 2 and (rank(data) - 1) not in axes and (-1) not in axes`.
*   **3**: `signal_size` - 1D tensor of type *T_SIZE* describing signal size with respect to axes from the input `axes`. If `signal_size[i] == -1`, then IRFFT is calculated for full size of the axis `axes[i]`. If `signal_size[i] > input_shape[: r - 1][axes[i]]`, then input data are zero-padded with respect to the axis `axes[i]` at the end. Finally, if `signal_size[i] < input_shape[: r - 1][axes[i]]`, then input data are trimmed with respect to the axis `axes[i]`. More precisely, if `signal_size[i] < input_shape[: r - 1][axes[i]]`, the slice `0: signal_size[i]` of the axis `axes[i]` is considered. Optional, with default value `[input_shape[: r - 1][a] for a in axes]`.
*   **NOTE**: If the input `signal_size` is specified, then the size of `signal_size` must be the same as the size of `axes`.

**Outputs**

*   **1**: Resulting tensor with elements of the same type as input `data` tensor. The shape of the output is calculated as follows. If the input `signal_size` is not specified, then the shape of output is the same as the shape of `data`. Otherwise, `output_shape[axis] = input_shape[axis]` for `axis not in axes`, and if `signal_size[i] == -1`, then `output_shape[: r - 1][axes[i]] = input_shape[: r - 1][axes[i]]`, else `output_shape[: r - 1][axes[i]] = signal_size[i]`.

**Types**

* *T*: floating-point type.

* *T_IND*: `int64` or `int32`.

* *T_SIZE*: `int64` or `int32`.
