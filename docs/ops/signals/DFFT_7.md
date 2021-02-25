## DFFT <a name="DFFT"></a> {#openvino_docs_ops_signals_DFFT_7}

**Versioned name**: *DFFT-7*

**Category**: Image processing

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
        pads_begin = np.zeros(self.input_rank).astype(np.int64)
        pads_end = np.zeros(self.input_rank).astype(np.int64)

        for a in self.axes:
            if self.output_shape[a] > self.input_shape[a]:
                pads_end[a] = self.output_shape[a] - self.input_shape[a]

        pads = list(zip(pads_begin, pads_end))
        padded_data = np.pad(self.data, pads, 'constant')
        slices = tuple([slice(0, d, 1) for d in self.output_shape])
        corrected_data = padded_data[slices]

        return corrected_data.copy()

    def __call__(self):
        corrected_data = self._get_corrected_data()
        # axes_and_lengths = sorted([(a, self.output_shape[a]) for a in self.axes], key=lambda p: p[0], reverse=True)

        reversed_input_shape = self.input_shape[::-1]
        reversed_output_shape = self.output_shape[::-1]

        input_strides = compute_strides(reversed_input_shape)
        output_strides = compute_strides(reversed_output_shape)

        result = np.zeros(list(self.output_shape))

        buffer_size = 0
        for axis in self.axes:
            current_length = self.output_shape[axis]
            size = 2 * current_length if is_power_of_two(x) else current_length
            buffer_size = max(buffer_size, size)

        buffer = np.zeros((buffer_size, 2))

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