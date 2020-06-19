## Interpolate <a name="Interpolate"></a>

**Versioned name**: *Interpolate-3*

**Category**: Image processing

**Short description**: *Interpolate* layer performs interpolation of independent slices in input tensor by specified dimensions and attributes.

**Attributes**

* *axes*

  * **Description**: `axes` specify dimension indices where interpolation is applied, and `axes` is any unordered list of indeces of different dimensions of input tensor, e.g. `[0, 4]`, `[4, 0]`, `[4, 2, 1]`, `[1, 2, 3]`. These indeces should be non-negative integers from `0` to `rank(data) - 1` inclusively.  Other dimensions do not change. The order of elements in `axes` attribute matters, and mapped directly to elements in the 2nd input `target_spatial_shape`. Namely, `output_shape[axes[i]] = target_spatial_shape[i]` for all `i in range(0, len(axes))` and `output_shape[j] = input_shape[j] + pads_begin[j] + pads_end[j]` for `j not in axes`, `j in range(0, rank(data))`.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

* *mode*

  * **Description**: specifies type of interpolation
  * **Range of values**: one of `nearest`, `linear`, `linear_onnx`, `cubic`, `area`
  * **Type**: string
  * **Default value**: none
  * **Required**: *yes*

* *coordinate_transformation_mode*

  * **Description**: specifies how to transform the coordinate in the resized tensor to the coordinate in the original tensor
  * **Range of values**: one of `half_pixel`, `pytorch_half_pixel`, `asymmetric`, `tf_half_pixel_for_nn`, `align_corners`
  * **Type**: string
  * **Default value**: `half_pixel`
  * **Required**: *no*

* *nearest_mode*

  * **Description**: specifies round mode when `mode == nearest` and is used only when `mode == nearest`.
  * **Range of values**: one of `round_prefer_floor`, `round_prefer_ceil`, `floor`, `ceil`, `simple`
  * **Type**: string
  * **Default value**: `round_prefer_floor`
  * **Required**: *no*

* *antialias*

  * **Description**: *antialias* is a flag that specifies whether to perform anti-aliasing.
  * **Range of values**:
    * False - do not perform anti-aliasing
    * True - perform anti-aliasing
  * **Type**: boolean
  * **Default value**: False
  * **Required**: *no*

* *pads_begin*

  * **Description**: *pads_begin* specifies the number of pixels to add to the beginning of the image being interpolated. This addition of pixels is done before interpolation calculation.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: `int[]`
  * **Default value**: `[0]`
  * **Required**: *no*

* *pads_end*

  * **Description**: *pads_end* specifies the number of pixels to add to the end of the image being interpolated. This addition of pixels is done before interpolation calculation.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: `int[]`
  * **Default value**: `[0]`
  * **Required**: *no*

* *cube_coeff*

* **Description**: *cube_coeff* specifies the parameter *a* for cubic interpolation (see, e.g.  [article](https://ieeexplore.ieee.org/document/1163711/)).  *cube_coeff* is used only when `mode == cubic`.
  * **Range of values**: floating point number
  * **Type**: any of supported floating point type
  * **Default value**: `-0.75`
  * **Required**: *no*

**Inputs**

*   **1**: `data` - Input tensor with data for interpolation. Type of elements is any supported type. Required.

*   **2**: `target_spatial_shape` - 1D tensor describing output shape for spatial axes. Number of elements matches the number of indices in *axes* attribute, the order matches as well. Required.

**Outputs**

*   **1**: Resulting interpolated tensor with elements of the same type as input `data` tensor. The shape of the output matches input `data` shape except spatial dimensions mentioned in `axes` attribute. For other dimensions shape matches sizes from `target_spaticl_shape` in order specified in `axes`.


**Detailed description**
Calculations are performed according to the following rules.

```python
import math
import numpy as np

class GetNearestPixel:
    def __init__(self, mode: str):
        self.func = {
            'round_prefer_floor': GetNearestPixel.prefer_floor_func,
            'round_prefer_ceil': GetNearestPixel.prefer_ceil_func,
            'floor': GetNearestPixel.floor_func,
            'ceil': GetNearestPixel.ceil_func,
            'simple': GetNearestPixel.simple_func
        }[mode]

    def __call__(self, x_original, is_downsample):
        return self.func(x_original, is_downsample)

    @staticmethod
    def prefer_floor_func(x_original, is_downsample):
        if x_original == int(x_original) + 0.5:
            return int(math.floor(x_original))
        else:
            return int(round(x_original))

    @staticmethod
    def prefer_ceil_func(x_original, is_downsample):
        return int(round(x_original))

    @staticmethod
    def floor_func(x_original, is_downsample):
        return int(math.floor(x_original))

    @staticmethod
    def ceil_func(x_original, is_downsample):
        return int(math.ceil(x_original))

    @staticmethod
    def simple_func(x_original, is_downsample):
        if is_downsample:
            return int(math.ceil(x_original))
        else:
            return int(x_original)


class GetOriginalCoordinate:
    def __init__(self, mode: str):
        self.func = {
            'half_pixel': GetOriginalCoordinate.half_pixel_func,
            'pytorch_half_pixel': GetOriginalCoordinate.pytorch_half_pixel_func,
            'asymmetric': GetOriginalCoordinate.asymmetric_func,
            'tf_half_pixel_for_nn': GetOriginalCoordinate.tf_half_pixel_for_nn_func,
            'align_corners': GetOriginalCoordinate.align_corners_func
        }[mode]

    def __call__(self, resized, x_scale, length_resized, length_original):
        return self.func(resized, x_scale, length_resized, length_original)

    @staticmethod
    def half_pixel_func(resized, x_scale, length_resized, length_original):
        return ((x_resized + 0.5) / x_scale) - 0.5

    @staticmethod
    def pytorch_half_pixel_func(resized, x_scale, length_resized, length_original):
        return (x_resized + 0.5) / x_scale - 0.5 if  length_resized > 1 else 0.0

    @staticmethod
    def asymmetric_func(resized, x_scale, length_resized, length_original):
        return x_resized / x_scale

    @staticmethod
    def tf_half_pixel_for_nn_func(resized, x_scale, length_resized, length_original):
        return (x_resized + 0.5) / x_scale

    @staticmethod
    def align_corners_func(resized, x_scale, length_resized, length_original):
        return length_resized == 1 ? 0 : x_resized * (length_original - 1) / (length_resized - 1)


def get_cubic_coeff(s, a):
    abs_s = abs(s)
    coeff = np.zeros(4)
    coeff[0] = a * (abs_s - 1.0) * (abs_s - 1.0) * abs_s
    coeff[1] = ((a + 2.0) * abs_s - (a + 3.0)) * abs_s * abs_s + 1.0
    coeff[2] = (((-a -2.0) * abs_s+ (2.0 * a + 3.0)) * abs_s - a) * abs_s
    coeff[3] = - a * abs_s * abs_s * (abs_s - 1.0)
    return coeff


def  triangle_coeffs(dz):
    return np.maximum(0.0, 1.0 - np.abs(dz))


class InterpolateCalculation:
    def __init__(self, attrs: dict):
        self.func = {
            'nearest': self.nearest_interpolation,
            'linear': self.linear_interpolation,
            'cubic': self.cubic_interpolation,
            'linear_onnx': self.onnx_linear_interpolation
        }['mode']

        if not('pads_begin' in attrs):
            self.pads_begin = [0]
        else:
            self.pads_begin = attrs['pads_begin']

        if not('pads_end' in attrs):
            self.pads_end = [0]
        else:
            self.pads_end = attrs['pads_end']

        if not ('coordinate_transformation_mode' in attrs):
            self.coordinate_transformation_mode = 'half_pixel'
        else:
            self.coordinate_transformation_mode = attrs['coordinate_transformation_mode']

        if ('align_corners' in attrs) and attrs['align_corners']:
            self.coordinate_transformation_mode = 'align_corners'

        if not ('nearest_mode' in attrs):
            self.nearest_mode = 'round_prefer_floor'
        else:
            self.nearest_mode = attrs['nearest_mode']

        if not ('cube_coeff' in attrs):
            self.cube_coeff = -0.75
        else:
            self.cube_coeff = attrs['cube_coeff']

        if not ('antialias' in self.attrs):
            self.antialias = False
        else:
            self.antialias = attrs['antialias']

        self.axes = np.array(attrs['axes']).astype(np.int64)
        self.get_original_coordinate = self.get_coordinate_transformation_mode()


    def get_coordinate_transformation_mode(self):
        return GetOriginalCoordinate(self.coordinate_transformation_mode)

    def shape_infer(self, input_data, target_spatial_shape):
        result = input_data.shape + self.pads_begin + self.pads_end
        for i in range(0, len(self.axes)):
            result[axes[i]] = target_spatial_shape[i]
        return result

    @staticmethod
    def correct_pad(pad, rank):
        pad_len = len(pad)
        if pad_len < rank:
            return np.pad(pad, (0, rank - pad_len)).astype(np.int64)
        elif pad_len > rank:
            return np.array(pad[: rank - 1]).astype(np.int64)
        else:
            return np.array(pad, dtype=np.int64)

    def __call__(self, input_data, target_spatial_shape):
        rank = input_data.ndim
        self.pads_begin = InterpolateCalculation.correct_pad(self.pads_begin, rank)
        self.pads_end = InterpolateCalculation.correct_pad(self.pads_end, rank)
        self.pads = list(zip(self.pads_begin, self.pads_end))

        self.output_shape = self.shape_infer(input_data, target_spatial_shape)
        padded_data = np.pad(input_data, self.pads)
        self.scales = self.output_shape / padded_data.shape
        self.input_shape = padded_data.shape
        return self.func(padded_data)

    def clip_coord(self, coord, axis):
        return max(0, min(coord, self.input_shape[axis] - 1))

    def cubic_interpolation(self, input_data):
        result = np.zeros(self.output_shape)
        num_of_axes = len(axes)
        indeces = np.ndindex(tuple(4 for _ in range(num_of_axes)))
        for coordinates in np.ndindex(self.output_shape):
            for index in indeces:
                input_coords = np.array(coordinates, dtype=np.int64)
                cubic_coeffs = []
                for i in range(len(index)):
                    axis = self.axes[i]
                    in_coord = self.get_original_coordinate(coordinates[axis], self.scales[axis], self.output_shape[axis], self.input_shape[axis])
                    cubic_coeffs.append(get_cubic_coeff(in_coord - math.floor(in_coord), self.cube_coeff))
                    input_coords[axis] = self.clip_coord(input_coords[axis] + index[i] - 1)
                data = input_data[input_coords]
                for i in range(len(index)):
                    data = data * cubic_coeffs[i][index[i]]
                result[coordinates] += data
        return result

    def linear_interpolation(self, input_data):
        result = np.zeros(self.output_shape)
        num_of_axes = len(axes)
        is_downsample = False

        for i in range(num_of_axes):
            is_downsample = is_downsample or (self.scales[axes[i]] < 1)

        antialias = is_downsample and self.antialias

        a = np.zeros(num_of_axes)
        for i in range(num_of_axes):
            a[i] = self.scales[axes[i]] if antialias else 1.0

        prod_of_a = np.prod(a)
        r = np.zeros(num_of_axes).astype(np.int64)
        for i in range(num_of_axes):
            r[i] = 2 if self.scales[axes[i]] > 1.0 else int(math.ceil(2.0/a[i]))

        indeces = np.ndindex(2 * r + 1)

        for coordinates in np.ndindex(self.output_shape):
            sum = 0
            wsum = 0

            icoords = np.array(coordinates).astype(np.float64)
            for i in range(num_of_axes):
                axis = axes[i]
                in_coord = self.get_original_coordinate(coordinates[axis],  self.scales[axis], self.output_shape[axis], self.input_shape[axis])
                icoords[axis] = in_coord
            icoords_r = np.around(icoords).astype(np.int64)

            for index in indeces:
                iarray = np.array(index).astype(np.int64) - r + input_coords[axes]
                conditions = [iarray[i] >= 0 and iarray[i] < self.input_shape[axes[i]] for i in range(num_of_axes)]
                if not all(conditions):
                    continue

                dz = icoords[axes] - iarray
                w = prod_of_a * np.prod(triangle_coeffs(dz))
                wsum += w
                input_indeces = np.array(coordinates).astype
                input_indeces[axes] = iarray
                sum += w * input_data[input_indeces]

            result[coordinates] = sum / wsum

        return result

    def onnx_linear_interpolation(self, input_data):
        rank = len(self.input_shape)
        assert rank in [2, 4], "mode 'linear_onnx' supports only 2D or 4D tensors"
        assert set(self.axes) == {2, 3} or set(self.axes) == {0, 1}, \
            "mode 'linear_onnx' supports only case when axes = {2, 3} or axes = {0, 1}"

        result = np.zeros(self.output_shape)

        if rank == 2:
            reshaped_data = np.reshape(input_data, (1, 1, self.input_shape[0], self.input_shape[1]))
            result = np.reshape(result,  (1, 1, self.output_shape[0], self.output_shape[1]))
        else:
            reshaped_data = input_data

        output_height = self.output_shape[0] if rank == 2 else self.output_shape[2]
        output_width = self.output_shape[1] if rank == 2 else self.output_shape[3]
        input_height = self.input_shape[0] if rank == 2 else self.input_shape[2]
        input_width = self.input_shape[1] if rank == 2 else self.input_shape[3]
        height_scale = self.scales[0] if rank == 2 else self.scales[2]
        width_scale = self.scales[1] if rank == 2 else self.scales[3]
        batch_size = 1 if rank == 2 else self.input_shape[0]
        num_channels = 1 if rank == 2 else self.input_shape[1]

        in_y1 = np.zeros(output_height).astype(np.int64)
        in_y2 = np.zeros(output_height).astype(np.int64)
        in_x1 = np.zeros(output_width).astype(np.int64)
        in_x2 = np.zeros(output_width).astype(np.int64)

        dy1 = np.zeros(output_height).astype(np.float64)
        dy2 = np.zeros(output_height).astype(np.float64)
        dx1 = np.zeros(output_width).astype(np.float64)
        dx2 = np.zeros(output_width).astype(np.float64)

        y_original = np.zeros(output_height).astype(np.float64)
        x_original = np.zeros(output_width).astype(np.float64)

        for y in range(output_height):
            in_y = self.get_original_coordinate(y, height_scale, output_height, input_height)
            y_original[y] = in_y
            in_y = max(0, min(in_y, input_height - 1))
            in_y1[y] = max(0, min(int(in_y), input_height - 1))
            in_y2[y] = min(in_y1[y] + 1, input_height - 1)
            dy1[y] = abs(in_y - in_y1[y])
            dy2[y] = abs(in_y - in_y2[y])

            if in_y1 == in_y2:
                dy1[y] = 0.5
                dy2[y] = 0.5

        for x in range(output_width):
            in_x = self.get_original_coordinate(x, width_scale, output_width, input_width)
            x_original[x] = in_x
            in_x = max(0, min(in_x, input_width - 1))
            in_x1[x] = max(0, min(int(in_x), input_width - 1))
            in_x2[x] = min(in_x1[x] + 1, input_width - 1)
            dx1[x] = abs(in_x - in_x1[x])
            dx2[x] = abs(in_x - in_x2[x])

            if in_x1 == in_x2:
                dx1[x] = 0.5
                dx2[x] = 0.5

        for n in range(batch_size):
            for c in range(num_channels):
                for y in range(output_height):
                    for x in range(output_width):
                        x11 = reshaped_data[n, c, in_y1[y], in_x1[x]]
                        x21 = reshaped_data[n, c, in_y1[y], in_x2[x]]
                        x12 = reshaped_data[n, c, in_y2[y], in_x1[x]]
                        x22 = reshaped_data[n, c, in_y2[y], in_x2[x]]
                        temp = dx2[x] * dy2[y] * x11 + dx1[x] * dy2[y] * x21
                        temp += dx2[x] * dy1[y] * x12 + dx1[x] * dy1[y] * x22
                        result[n, c, y, x] = temp

        return np.reshape(result, self.output_shape)

    def nearest_interpolation(self, input_data):
        if not ('nearest_mode' in self.attrs):
            self.attrs['nearest_mode'] = 'floor'

        self.get_nearest_pixel = GetNearestPixel(attrs['nearest_mode'])

        result = np.zeros(self.output_shape)

        num_of_axes = len(axes)
        for coordinates in np.ndindex(self.output_shape):
            input_coords = np.array(coordinates, dtype=np.int64)
            for i in range(num_of_axes):
                axis = axes[i]
                in_coord = self.get_original_coordinate(coordinates[axis], self.scales[axis], self.output_shape[axis], self.input_shape[axis])
                nearest_pixel = self.get_nearest_pixel(in_coord, self.scales[axis] < 1)
                input_coords[axis] = max(0, min(nearest_pixel, self.input_shape[axis] - 1))
           result[coordinates] = input_data[input_coords]

        return result
```


**Example**

```xml
<layer ... type="Interpolate" ...>
    <data axes="2,3" align_corners="0" pads_begin="0" pads_end="0" mode="linear"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>2</dim>
            <dim>48</dim>
            <dim>80</dim>
        </port>
        <port id="1">
            <dim>2</dim>  <!--The values in this input are [50, 60] -->
        </port>
    </input>
    <output>
        <port id="0">
            <dim>1</dim>
            <dim>2</dim>
            <dim>50</dim>
            <dim>60</dim>
        </port>
    </output>
</layer>
```
