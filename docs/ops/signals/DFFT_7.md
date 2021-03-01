## DFFT <a name="DFFT"></a> {#openvino_docs_ops_signals_DFFT_7}

**Versioned name**: *DFFT-7*

**Category**: Signal processing

**Short description**: *DFFT* layer performs the discrete fast Fourier transformation of input tensor by specified dimensions.

**Attributes**:

    No attributes available.

**Inputs**

*   **1**: `data` - Input tensor of type *T* with data for the DFFT transformation. Type of elements is any supported floating point type or `int8` type. Required.
*   **2**: `axes` - 1D tensor of type *T_IND* specifying dimension indices where DFFT is applied, and `axes` is any unordered list of indices of different dimensions of input tensor, e.g. `[0, 4]`, `[4, 0]`, `[4, 2, 1]`, `[1, 2, 3]`. These indices should be non-negative integers from `0` to `rank(data) - 1` inclusively.  Other dimensions do not change. The order of elements in `axes` attribute matters, and mapped directly to elements in the 3d input `signal_size`. Required.
*   **3**: `signal_size` - 1D tensor of type *T_IND* describing signal size with respect to axes from the input `axes`. For any `i in range(0, len(axes))`, if `signal_size[i] == -1`, then DFFT is calculated for full size of the axis `axes[i]`. If `signal_size[i] > input_shape[axes[i]]`, then input data are zero-padded with respect to the axis `axes[i]`. Finally, `signal_size[i] < input_shape[axes[i]]`, then input data are trimmed with respect to the axis `axes[i]`. Optional, with default value `[input_shape[a] for a in axes]`.
*   **Note**: The following constraint must be satisfied: `rank(data) >= len(axes) + 1 and input_shape[-1] == 2 and (rank(data) - 1) not in axes`.

**Outputs**

*   **1**: Resulting tensor with elements of the same type as input `data` tensor. The shape of the output matches input `data` shape except dimensions mentioned in `axes` input. For other dimensions shape matches sizes from `signal_size` in order specified in `axes`.

**Types**

* *T*: floating point type.

* *T_IND*: `int64` or `int32`.

**Detailed description**: *DFFT* performs the discrete fast Fourier transformation of input tensor with respect to specified axes. Calculations are performed according to the following rules.

```python
import cmath
import math
import numpy as np


def compute_strides(arr):
    strides = np.zeros(len(arr) + 1).astype(np.int64)
    stride = 1
    for i in range(0, len(arr)):
        strides[i] = stride
        stride *= arr[i]

    strides[-1] = stride
    return strides


def is_power_of_two(x):
    return (x != 0) and ((x & (x - 1)) == 0)


def simple_get_data_from_input(input_data, src_index, fft_size, fft_lengths, fft_strides):
    num_strides = len(fft_strides)
    return input_data[src_index : src_index + fft_size]


def generic_get_data_from_input(input_data, start_index, fft_size, fft_lengths, fft_strides):
    result = np.zeros(fft_size).astype(np,complex64)

    fft_rank = len(fft_lengths)
    input_coords = np.zeros(fft_rank, dtype=np.int64)

    for i in range(0, fft_size):
        curr = i
        for j in list(range(0, fft_rank))[::-1]:
            input_coords[j] = curr // fft_lengths[j]
            curr %= fft_lengths[j]
        input_coords[0] = curr

        offset = 0
        for j in range(0, fft_rank):
            offset += input_coords[j] * fft_strides[j]

        result[i] = input_data[start_index + offset]
    return result


def twiddle(k, length):
    return cmath.exp(complex(0.0, -2.0 * math.pi * k / length))


class DFFT:
    def __init__(self, data, axes, signal_size=None):
        assert len(set(axes)) != len(axes), "DFFT doesn't support non-unique axes. Got: {0}".format(axes)

        self.data = data
        self.axes = axes
        self.input_shape = np.array(data.shape, dtype=np.int64)
        self.fft_rank = len(axes)

        assert self.input_rank >= self.fft_rank + 1, \
            "Input rank must be greater than number of axes. "
            "Got: input rank = {0}, number of axes = {1}".format(self.input_rank, self.fft_rank)

        assert (self.input_rank - 1) not in axes, "Axis for DFFT must not be the last axis. Got axes: {}".format(axes)

        assert self.input_shape[-1] == 2, \
            "The last dimension of input data must be equal to 2. Got input shape: {}".format(self.input_shape)

        self.signal_size = signal_size if signal_size is not None else [self.input_shape[a] for a in axes]
        self.input_rank = len(self.input_shape)

    def _shape_infer(self):
        output_shape = self.input_shape.copy()
        for a in self.axes:
            if self.signal_size[a] == -1:
                continue
            output_shape[a] = self.signal_size[a]

        self.output_shape = output_shape.copy()

    def _get_corrected_data(self):
        self._shape_infer()
        pads_begin = np.zeros(self.input_rank).astype(np.int64)
        pads_end = np.zeros(self.input_rank).astype(np.int64)

        for a in self.axes:
            if self.output_shape[a] > self.input_shape[a]:
                pads_end[a] = self.output_shape[a] - self.input_shape[a]

        pads = list(zip(pads_begin, pads_end))
        padded_data = np.pad(self.data, pads, 'constant')
        slices = tuple([slice(0, d, 1) for d in self.output_shape])
        corrected_data = np.squeeze(np.view(padded_data[slices], dtype=np.complex64), axis=-1)

        return corrected_data.copy()

    def __call__(self):
        corrected_data = self._get_corrected_data()
        corrected_data_rank = self.input_rank - 1 # rank of complex tensor
        flattened_data = np.ravel(corrected_data)

        reversed_shape = self.output_shape[::-1][1:] # if real tensor output_shape was [12, 152, 314, 1936, 87, 2], then
                                                     # we have [87, 1936, 314, 152, 12] as reversed shape of
                                                     # complex tensor
        strides = compute_strides(reversed_shape)
        all_axes = np.arange(0, corrected_data_rank).astype(np.int64)

        sorted_axes = sorted(self.axes, reverse=True)
        fft_axes = np.array([corrected_data_rank - 1 - a for a in sorted_axes], dtype=np.int64)

        fft_lengths = np.array([reversed_shape[a] for a in fft_axes], dtype=np.int64)
        fft_strides = np.array([strides[a] for a in fft_axes], dtype=np.int64)
        fft_size = np.prod(fft_lengths)

        result = np.zeros(list(self.output_shape)).astype(np.complex64)
        flattened_result = np.ravel(result)

        if fft_size <= 0:
            return result

        bits_for_fft_axes = 0
        for axis in fft_axes:
            bits_for_fft_axes |= 1 << axis

        outer_axes = []
        for i in range(0, corrected_data_rank):
            if (bits_for_fft_axes & (1 << i)) == 0:
                outer_axes.append(i)
        outer_axes = np.array(outer_axes, dtype=np.int64)
        outer_rank = len(outer_axes)

        outer_axes_lengths = np.array([reversed_shape[a] for a in outer_axes], dtype=np.int64)
        outer_axes_strides = np.array([strides[a] for a in outer_axes], dtype=np.int64)
        outer_size = np.prod(outer_axes_lengths)

        data = np.zeros(fft_size).astype(np.complex64)

        buffer_size = 0
        for axis in reversed_axis:
            current_length = self.output_shape[axis]
            size = 2 * current_length if is_power_of_two(x) else current_length
            buffer_size = max(buffer_size, size)
        buffer = np.zeros(buffer_size).astype(np.complex64)

        if sorted(reversed_axes) == list(range(0, fft_rank)):
            simple_axes = True
            self.get_data_from_input_func = simple_get_data_from_input
        else:
            simple_axes = False
            self.get_data_from_input_func = generic_get_data_from_input

        outer_coords = np.zeros(outer_rank).astype(np.int64)
        for outer_idx in range(0, outer_size):
            curr = outer_idx
            for j in list(range(0, outer_rank))[::-1]:
                outer_coords[j] = curr // outer_axes_lengths[j]
                curr %= outer_axes_lengths[j]
            outer_coords[0] = curr

            index = 0
            for j in range(0, outer_rank):
                index += outer_coords[j] * outer_axes_strides[j]

            data = self.get_data_from_input_func(flattened_data, src_index, fft_size, fft_lengths, fft_strides)

            for idx, axis in enumerate(fft_axes):
                # TODO: write calculation with respect to current axis
                pass

            # Copying of calculation result
            if simple_axes:
                flattened_result[index: index + fft_size] = data
            else:
                fft_coords_to_write = np.zeros(fft_rank, dtype=np.int64)
                for i in range(0, fft_size):
                    current = i
                    for j in list(range(0, fft_rank))[::-1]:
                        fft_coords_to_write[j] = current // fft_lengths[j]
                        current %= fft_lengths[j]
                    fft_coords_to_write[0] = current

                    offset = 0
                    for j in range(0, fft_rank):
                        offset += fft_coords_to_write[j] * fft_strides[j]

                    flattened_result[index + offset] = data[offset]

        return result
```

**Example**:

There is no `signal_size` input:
```xml
<layer ... type="DFFT" ... >
    <data normalization_mode="forward"/>
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
<layer ... type="DFFT" ... >
    <data normalization_mode="forward"/>
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