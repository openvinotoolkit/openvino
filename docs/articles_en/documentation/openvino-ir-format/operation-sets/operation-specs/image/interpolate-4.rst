Interpolate
===========


.. meta::
  :description: Learn about Interpolate-4 - an image processing operation, which
                can be performed on three required and one optional tensor.

**Versioned name**: *Interpolate-4*

**Category**: *Image processing*

**Short description**: *Interpolate* layer performs interpolation of independent slices in input tensor by specified dimensions and attributes.

**Attributes**

* *mode*

  * **Description**: specifies type of interpolation
  * **Range of values**: one of ``nearest``, ``linear``, ``linear_onnx``, ``cubic``
  * **Type**: string
  * **Required**: *yes*

    .. note::

       Only 2D, 3D, 4D, 5D tensors with ``axes = {0, 1}``, ``axes = {0, 1, 2}``,
       ``axes = {2, 3}``,  ``axes = {2, 3, 4}`` respectively are supported for
       ``"mode" == "linear_onnx"``.

* *shape_calculation_mode*

  * **Description**: specifies which input, ``sizes`` or ``scales``, is used to calculate an output shape.
  * **Range of values**: name of a shape calculation mode in string format:

    * ``sizes`` - an output shape is calculated as ``output_shape[axes[i]] = sizes[i]`` for all ``i in range(0, len(axes))`` and ``output_shape[j] = input_shape[j] + pads_begin[j] + pads_end[j]`` for ``j not in axes``, ``j in range(0, rank(data))``.
    * ``scales`` - an output shape is calculated as ``output_shape[axes[i]] = floor(scales[i] * (input_shape[axes[i]] + pads_begin[axes[i]] + pads_end[axes[i]]))`` for all ``i in range(0, len(axes))`` and ``output_shape[j] = input_shape[j] + pads_begin[j] + pads_end[j]`` for ``j not in axes``, ``j in range(0, rank(data))``

  * **Type**: string
  * **Required**: *yes*

* *coordinate_transformation_mode*

  * **Description**: specifies how to transform the coordinate in the resized tensor to the coordinate in the original tensor
  * **Range of values**: name of the transformation mode in string format (here ``scale[x]`` is ``output_shape[x] / input_shape[x]`` and ``x_resized`` is a coordinate in axis ``x``, for any axis ``x`` from the input ``axes``):

    * ``half_pixel`` - the coordinate in the original tensor axis ``x`` is calculated as ``((x_resized + 0.5) / scale[x]) - 0.5``.
    * ``pytorch_half_pixel`` -  the coordinate in the original tensor axis ``x`` is calculated by ``(x_resized + 0.5) / scale[x] - 0.5 if  output_shape[x] > 1 else 0.0``.
    * ``asymmetric`` - the coordinate in the original tensor axis ``x`` is calculated according to the formula ``x_resized / scale[x]``.
    * ``tf_half_pixel_for_nn`` - the coordinate in the original tensor axis ``x`` is ``(x_resized + 0.5) / scale[x]``.
    * ``align_corners`` - the coordinate in the original tensor axis ``x`` is calculated as ``0 if output_shape[x] == 1 else  x_resized * (input_shape[x] - 1) / (output_shape[x] - 1)``.

  * **Type**: string
  * **Default value**: ``half_pixel``
  * **Required**: *no*

* *nearest_mode*

  * **Description**: specifies round mode when ``mode == nearest`` and is used only when ``mode == nearest``.
  * **Range of values**: name of the round mode in string format:
    * ``round_prefer_floor`` - this mode is known as round half down.
    * ``round_prefer_ceil`` - it is round half up mode.
    * ``floor`` - this mode computes the largest integer value not greater than rounded value.
    * ``ceil`` - this mode computes the smallest integer value not less than rounded value.
    * ``simple`` - this mode behaves as ``ceil`` mode when ``Interpolate`` is downsample, and as dropping the fractional part otherwise.
  * **Type**: string
  * **Default value**: ``round_prefer_floor``
  * **Required**: *no*

* *antialias*

  * **Description**: *antialias* is a flag that specifies whether to perform anti-aliasing.
  * **Range of values**:
    * false - do not perform anti-aliasing
    * true - perform anti-aliasing
  * **Type**: boolean
  * **Default value**: false
  * **Required**: *no*

* *pads_begin*

  * **Description**: *pads_begin* specifies the number of pixels to add to the beginning of the image being interpolated. This addition of pixels is done before interpolation calculation.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: ``int[]``
  * **Default value**: ``[0]``
  * **Required**: *no*

* *pads_end*

  * **Description**: *pads_end* specifies the number of pixels to add to the end of the image being interpolated. This addition of pixels is done before interpolation calculation.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: ``int[]``
  * **Default value**: ``[0]``
  * **Required**: *no*

* *cube_coeff*

  * **Description**: *cube_coeff* specifies the parameter *a* for cubic interpolation (see, e.g. `article <https://ieeexplore.ieee.org/document/1163711/>`__ ).  *cube_coeff* is used only when ``mode == cubic``.
  * **Range of values**: floating-point number
  * **Type**: any of supported floating-point type
  * **Default value**: ``-0.75``
  * **Required**: *no*

**Inputs**

* **1**: ``data`` - tensor of type *T* with data for interpolation. **Required.**

* **2**: ``sizes`` - 1D tensor of type *T_SIZE* describing output shape for spatial axes. Number of elements matches the number of indices in ``axes`` input, the order matches as well. **Required.**

* **3**: ``scales`` - 1D tensor of type *T_SCALES* describing scales for spatial axes. Number and order of elements match the number and order of indices in ``axes`` input. **Required.**

* **4**: ``axes`` - 1D tensor of type *T_AXES* specifying dimension indices where interpolation is applied, and ``axes`` is any unordered list of indices of different dimensions of input tensor, e.g. ``[0, 4]``, ``[4, 0]``, ``[4, 2, 1]``, ``[1, 2, 3]``. These indices should be non-negative integers from ``0`` to ``rank(data) - 1`` inclusively.  Other dimensions do not change. The order of elements in ``axes`` attribute matters, and mapped directly to elements in the 2nd input ``sizes``. **Optional** with default value ``[0,...,rank(data) - 1]``.

**Outputs**

* **1**: Resulting interpolated tensor with elements of the same type as input ``data`` tensor. The shape of the output matches input ``data`` shape except spatial dimensions mentioned in ``axes`` attribute. For other dimensions shape matches sizes from ``sizes`` in order specified in ``axes``.

**Types**

* *T*: any supported numeric type.
* *T_SIZE*: any supported integer type.
* *T_SCALES*: any supported floating-point type.
* *T_AXES*: any supported integer type.


**Detailed description**
Calculations are performed according to the following rules.

.. code-block:: py
   :force:

   import math
   import numpy as np
   from enum import Enum, unique

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

       def __call__(self, x_resized, x_scale, length_resized, length_original):
           return self.func(x_resized, x_scale, length_resized, length_original)

       @staticmethod
       def half_pixel_func(x_resized, x_scale, length_resized, length_original):
           return ((x_resized + 0.5) / x_scale) - 0.5

       @staticmethod
       def pytorch_half_pixel_func(x_resized, x_scale, length_resized, length_original):
           return (x_resized + 0.5) / x_scale - 0.5 if length_resized > 1 else 0.0

       @staticmethod
       def asymmetric_func(x_resized, x_scale, length_resized, length_original):
           return x_resized / x_scale

       @staticmethod
       def tf_half_pixel_for_nn_func(x_resized, x_scale, length_resized, length_original):
           return (x_resized + 0.5) / x_scale

       @staticmethod
       def align_corners_func(x_resized, x_scale, length_resized, length_original):
           return  0 if length_resized == 1 else  x_resized * (length_original - 1) / (length_resized - 1)


   def get_cubic_coeff(s, a):
       abs_s = abs(s)
       coeff = np.zeros(4)
       coeff[0] = a * (abs_s - 1.0) * (abs_s - 1.0) * abs_s
       coeff[1] = ((a + 2.0) * abs_s - (a + 3.0)) * abs_s * abs_s + 1.0
       coeff[2] = (((-a -2.0) * abs_s+ (2.0 * a + 3.0)) * abs_s - a) * abs_s
       coeff[3] = - a * abs_s * abs_s * (abs_s - 1.0)
       return coeff


   def triangle_coeffs(dz):
       return np.maximum(0.0, 1.0 - np.abs(dz))


   @unique
   class ShapeCalculationMode(Enum):
       SIZES = 0
       SCALES = 1


   class InterpolateCalculation:
       def __init__(self, attrs: dict):
           self.mode = attrs['mode']
           self.func = {
               'nearest': self.nearest_interpolation,
               'linear': self.linear_interpolation,
               'cubic': self.cubic_interpolation,
               'linear_onnx': self.onnx_linear_interpolation
           }[self.mode]
           self.attrs = attrs

           self.pads_begin = attrs.get('pads_begin', [0])
           self.pads_end = attrs.get('pads_end', [0])
           self.coordinate_transformation_mode = attrs.get('coordinate_transformation_mode', 'half_pixel')
           self.nearest_mode = attrs.get('nearest_mode', 'round_prefer_floor')
           self.cube_coeff = attrs.get('cube_coeff', -0.75)
           self.antialias = attrs.get('antialias', False)

           self.shape_calculation_mode = {
               'sizes': ShapeCalculationMode.SIZES,
               'scales': ShapeCalculationMode.SCALES
           }[attrs['shape_calculation_mode']]

           self.get_original_coordinate = self.get_coordinate_transformation_mode()
           self.get_nearest_pixel = GetNearestPixel(self.nearest_mode)


       def get_coordinate_transformation_mode(self):
           return GetOriginalCoordinate(self.coordinate_transformation_mode)

       def shape_infer(self, input_data, sizes, scales):
           result = input_data.shape + self.pads_begin + self.pads_end

           if self.shape_calculation_mode == ShapeCalculationMode.SIZES:
               for i, axis in enumerate(self.axes):
                   result[axis] = sizes[i]
           else:
               for i, axis in enumerate(self.axes):
                   result[axis] = math.floor(scales[i] * result[axis])

           return result

       @staticmethod
       def correct_pad(pad, rank):
           pad_len = len(pad)
           if pad_len < rank:
               return np.pad(pad, (0, rank - pad_len), 'constant').astype(np.int64)
           elif pad_len > rank:
               return np.array(pad[: rank - 1]).astype(np.int64)
           else:
               return np.array(pad, dtype=np.int64)

       def __call__(self, input_data, sizes, scales, axes):
           rank = input_data.ndim
           self.pads_begin = InterpolateCalculation.correct_pad(self.pads_begin, rank)
           self.pads_end = InterpolateCalculation.correct_pad(self.pads_end, rank)
           self.pads = list(zip(self.pads_begin, self.pads_end))
           self.axes = np.array(axes).astype(np.int64)

           self.output_shape = self.shape_infer(input_data, sizes, scales)
           padded_data = np.pad(input_data, self.pads, 'constant')

           if self.shape_calculation_mode == ShapeCalculationMode.SIZES:
               num_of_axes = len(self.axes)
               self.scales = np.zeros(num_of_axes)
               for i, axis in enumerate(axes):
                   self.scales[i] = self.output_shape[axis] / padded_data.shape[axis]
           else:
               self.scales = scales

           if self.mode == 'nearest':
               self.all_scales = np.ones(rank).astype(np.float)
               for i, axis in enumerate(self.axes):
                   self.all_scales[axis] = self.scales[i]

           self.input_shape = padded_data.shape
           return self.func(padded_data)

       def clip_coord(self, coord, axis):
           return max(0, min(coord, self.input_shape[axis] - 1))

       def cubic_interpolation(self, input_data):
           rank = len(self.input_shape)
           result = np.zeros(self.output_shape)
           num_of_axes = len(self.axes)
           indices = [ind for ind in np.ndindex(tuple(4 for _ in range(num_of_axes)))]
           for coordinates in np.ndindex(tuple(self.output_shape)):
               input_coords = np.array(coordinates, dtype=np.int64)
               cubic_coeffs = np.zeros((rank, 4))
               for i, axis in enumerate(self.axes):
                   in_coord = self.get_original_coordinate(coordinates[axis], self.scales[i], self.output_shape[axis], self.input_shape[axis])
                   in_coord_int = math.floor(in_coord)
                   input_coords[axis] = in_coord_int
                   cubic_coeffs[axis] = get_cubic_coeff(in_coord - in_coord_int, self.cube_coeff)
               summa = 0.0
               for index in indices:
                   coords_for_sum = input_coords.copy()
                   coeffs_prod = 1.0
                   for i, axis in enumerate(self.axes):
                       coords_for_sum[axis] = self.clip_coord(input_coords[axis] + index[i] - 1, axis)
                   for i, axis in enumerate(self.axes):
                       coeffs_prod = coeffs_prod * cubic_coeffs[axis][index[i]]
                   summa += coeffs_prod * input_data[tuple(coords_for_sum)]
               result[coordinates] = summa
           return result

       def linear_interpolation(self, input_data):
           result = np.zeros(self.output_shape)
           num_of_axes = len(self.axes)
           is_downsample = False

           for scale in self.scales:
               is_downsample = is_downsample or (scale < 1)

           antialias = is_downsample and self.antialias

           a = np.zeros(num_of_axes)
           for i, _ in enumerate(self.axes):
               a[i] = self.scales[i] if antialias else 1.0

           prod_of_a = np.prod(a)
           r = np.zeros(num_of_axes).astype(np.int64)
           for i, _ in enumerate(self.axes):
               r[i] = 2 if self.scales[i] > 1.0 else int(math.ceil(2.0/a[i]))

           indices = [tuple(np.array(ind).astype(np.int64) - r) for ind in np.ndindex(tuple(2 * r + 1))]

           for coordinates in np.ndindex(tuple(self.output_shape)):
               icoords = np.array(coordinates).astype(np.float64)
               icoords_r = np.array(coordinates).astype(np.float64)
               for i, axis in enumerate(self.axes):
                   in_coord = self.get_original_coordinate(coordinates[axis], self.scales[i], self.output_shape[axis], self.input_shape[axis])
                   icoords[axis] = in_coord
                   icoords_r[axis] = round(in_coord)

               summa = 0.0
               wsum = 0.0

               for index in indices:
                   inner_coords = np.array(coordinates)
                   for i, axis in enumerate(self.axes):
                       inner_coords[axis] = index[i] + icoords_r[axis]

                   conditions = [inner_coords[axis] >= 0 and inner_coords[axis] < self.input_shape[axis] for axis in self.axes]
                   if not all(conditions):
                       continue

                   dz = np.zeros(num_of_axes)
                   for i, axis in enumerate(self.axes):
                       dz[i] = icoords[axis] - inner_coords[axis]

                   w = prod_of_a * np.prod(triangle_coeffs(a * dz))
                   wsum += w
                   summa += w * input_data[tuple(inner_coords)]

               if wsum == 0:
                   result[coordinates] = 0.0
               else:
                   result[coordinates] = summa / wsum

           return result

       def onnx_linear_interpolation5D(self, input_data):
           rank = len(self.input_shape)
           assert rank in [3, 5], "mode 'linear_onnx' supports only 3D or 5D tensors"
           assert set(self.axes) == {2, 3, 4} or set(self.axes) == {0, 1, 2}, \
               "mode 'linear_onnx' supports only case when axes = {2, 3, 4} or axes = {0, 1, 2}"

           result = np.zeros(self.output_shape)

           if rank == 3:
               reshaped_data = np.reshape(input_data, (1, 1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
               result = np.reshape(result,  (1, 1, self.output_shape[0], self.output_shape[1], self.output_shape[2]))
           else:
               reshaped_data = input_data

           input_shape = np.array(reshaped_data.shape).astype(np.int64)
           output_shape = np.array(result.shape).astype(np.int64)

           batch_size = input_shape[0];
           num_channels = input_shape[1];
           input_depth = input_shape[2];
           input_height = input_shape[3];
           input_width = input_shape[4];
           output_depth = output_shape[2];
           output_height = output_shape[3];
           output_width = output_shape[4];

           depth_scale = self.scales[0];
           height_scale = self.scales[1];
           width_scale = self.scales[2];

           z_original = np.zeros(output_depth).astype(np.float)
           y_original = np.zeros(output_height).astype(np.float)
           x_original = np.zeros(output_width).astype(np.float)

           in_z1 = np.zeros(output_depth).astype(np.int64)
           in_z2 = np.zeros(output_depth).astype(np.int64)
           in_y1 = np.zeros(output_height).astype(np.int64)
           in_y2 = np.zeros(output_height).astype(np.int64)
           in_x1 = np.zeros(output_width).astype(np.int64)
           in_x2 = np.zeros(output_width).astype(np.int64)

           dz1 = np.zeros(output_depth).astype(np.float)
           dz2 = np.zeros(output_depth).astype(np.float)

           dy1 = np.zeros(output_height).astype(np.float)
           dy2 = np.zeros(output_height).astype(np.float)

           dx1 = np.zeros(output_width).astype(np.float)
           dx2 = np.zeros(output_width).astype(np.float)

           for z in range(0, output_depth):
               in_z = self.get_original_coordinate(z, depth_scale, output_depth, input_depth)
               z_original[z] = in_z
               in_z = max(0, min(in_z, input_depth - 1))
               in_z1[z] = max(0, min(int(in_z), input_depth - 1))
               in_z2[z] = min(in_z1[z] + 1, input_depth - 1)
               dz1[z] = abs(in_z - in_z1[z])
               dz2[z] = abs(in_z - in_z2[z])

               if in_z1[z] == in_z2[z]:
                   dz1[z] = 0.5
                   dz2[z] = 0.5

           for y in range(0, output_height):
               in_y = self.get_original_coordinate(y, height_scale, output_height, input_height)
               y_original[y] = in_y
               in_y = max(0, min(in_y, input_height - 1))
               in_y1[y] = max(0, min(int(in_y), input_height - 1))
               in_y2[y] = min(in_y1[y] + 1, input_height - 1)
               dy1[y] = abs(in_y - in_y1[y])
               dy2[y] = abs(in_y - in_y2[y])

               if in_y1[y] == in_y2[y]:
                   dy1[y] = 0.5
                   dy2[y] = 0.5

           for x in range(0, output_width):
               in_x = self.get_original_coordinate(x, width_scale, output_width, input_width);
               x_original[x] = in_x
               in_x = max(0.0, min(in_x, input_width - 1));

               in_x1[x] = min(in_x, input_width - 1);
               in_x2[x] = min(in_x1[x] + 1, input_width - 1);

               dx1[x] = abs(in_x - in_x1[x]);
               dx2[x] = abs(in_x - in_x2[x]);
               if in_x1[x] == in_x2[x]:
                   dx1[x] = 0.5
                   dx2[x] = 0.5
           for n in range(0, batch_size):
               for c in range(0, num_channels):
                   for z in range(0, output_depth):
                       for y in range(0, output_height):
                           for x in range(0, output_width):
                               x111 = reshaped_data[n, c, in_z1[z], in_y1[y], in_x1[x]]
                               x211 = reshaped_data[n, c, in_z1[z], in_y1[y], in_x2[x]]
                               x121 = reshaped_data[n, c, in_z1[z], in_y2[y], in_x1[x]]
                               x221 = reshaped_data[n, c, in_z1[z], in_y2[y], in_x2[x]]
                               x112 = reshaped_data[n, c, in_z2[z], in_y1[y], in_x1[x]]
                               x212 = reshaped_data[n, c, in_z2[z], in_y1[y], in_x2[x]]
                               x122 = reshaped_data[n, c, in_z2[z], in_y2[y], in_x1[x]]
                               x222 = reshaped_data[n, c, in_z2[z], in_y2[y], in_x2[x]]

                               temp = dx2[x] * dy2[y] * dz2[z] * x111 + dx1[x] * dy2[y] * dz2[z] * x211
                               temp += dx2[x] * dy1[y] * dz2[z] * x121 + dx1[x] * dy1[y] * dz2[z] * x221
                               temp += dx2[x] * dy2[y] * dz1[z] * x112 + dx1[x] * dy2[y] * dz1[z] * x212
                               temp += dx2[x] * dy1[y] * dz1[z] * x122 + dx1[x] * dy1[y] * dz1[z] * x222

                               result[n, c, z, y, x] = temp

           return np.reshape(result, self.output_shape)

       def onnx_linear_interpolation4D(self, input_data):
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

           input_shape = np.array(reshaped_data.shape).astype(np.int64)
           output_shape = np.array(result.shape).astype(np.int64)

           output_height = output_shape[2]
           output_width = output_shape[3]
           input_height = input_shape[2]
           input_width = input_shape[3]
           height_scale = self.scales[0]
           width_scale = self.scales[1]
           batch_size = input_shape[0]
           num_channels = input_shape[1]

           y_original = np.zeros(output_height).astype(np.float)
           x_original = np.zeros(output_width).astype(np.float)

           in_y1 = np.zeros(output_height).astype(np.int64)
           in_y2 = np.zeros(output_height).astype(np.int64)
           in_x1 = np.zeros(output_width).astype(np.int64)
           in_x2 = np.zeros(output_width).astype(np.int64)

           dy1 = np.zeros(output_height).astype(np.float)
           dy2 = np.zeros(output_height).astype(np.float)

           dx1 = np.zeros(output_width).astype(np.float)
           dx2 = np.zeros(output_width).astype(np.float)

           for y in range(0, output_height):
               in_y = self.get_original_coordinate(y, height_scale, output_height, input_height)
               y_original[y] = in_y
               in_y = max(0, min(in_y, input_height - 1))
               in_y1[y] = max(0, min(int(in_y), input_height - 1))
               in_y2[y] = min(in_y1[y] + 1, input_height - 1)
               dy1[y] = abs(in_y - in_y1[y])
               dy2[y] = abs(in_y - in_y2[y])

               if in_y1[y] == in_y2[y]:
                   dy1[y] = 0.5
                   dy2[y] = 0.5

           for x in range(0, output_width):
               in_x = self.get_original_coordinate(x, width_scale, output_width, input_width);
               x_original[x] = in_x
               in_x = max(0.0, min(in_x, input_width - 1));

               in_x1[x] = min(in_x, input_width - 1);
               in_x2[x] = min(in_x1[x] + 1, input_width - 1);

               dx1[x] = abs(in_x - in_x1[x]);
               dx2[x] = abs(in_x - in_x2[x]);
               if in_x1[x] == in_x2[x]:
                   dx1[x] = 0.5
                   dx2[x] = 0.5

           for n in range(0, batch_size):
               for c in range(0, num_channels):
                   for y in range(0, output_height):
                       for x in range(0, output_width):
                           x11 = reshaped_data[n, c, in_y1[y], in_x1[x]]
                           x21 = reshaped_data[n, c, in_y1[y], in_x2[x]]
                           x12 = reshaped_data[n, c, in_y2[y], in_x1[x]]
                           x22 = reshaped_data[n, c, in_y2[y], in_x2[x]]
                           temp = dx2[x] * dy2[y] * x11 + dx1[x] * dy2[y] * x21 + dx2[x] * dy1[y] * x12 + dx1[x] * dy1[y] * x22
                           result[n, c, y, x] = temp

           return np.reshape(result, self.output_shape)

       def onnx_linear_interpolation(self, input_data):
           rank = len(self.input_shape)
           assert rank in [2, 3, 4, 5], "mode 'linear_onnx' supports only 2D, 3D, 4D, or 5D tensors"

           if rank in [2, 4]:
               self.onnx_linear_interpolation4D(input_data)
           else:
               self.onnx_linear_interpolation5D(input_data)

       def nearest_interpolation(self, input_data):
           result = np.zeros(self.output_shape)

           num_of_axes = len(self.axes)
           for coordinates in np.ndindex(tuple(self.output_shape)):
               input_coords = np.array(coordinates, dtype=np.int64)
               for axis, scale in enumerate(self.all_scales):
                   in_coord = self.get_original_coordinate(coordinates[axis], scale, self.output_shape[axis], self.input_shape[axis])
                   nearest_pixel = self.get_nearest_pixel(in_coord, scale < 1)
                   input_coords[axis] = max(0, min(nearest_pixel, self.input_shape[axis] - 1))
               result[coordinates] = input_data[tuple(input_coords)]

           return result



**Example**

.. code-block:: xml
   :force:

   <layer ... type="Interpolate" ...>
       <data shape_calculation_mode="scales" pads_begin="0" pads_end="0" mode="linear"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>2</dim>
               <dim>48</dim>
               <dim>80</dim>
           </port>
           <port id="1">
               <dim>2</dim>  <!--The values in this input are [24, 160] -->
           </port>
           <port id="2">
               <dim>2</dim>  <!--The values in this input are [0.5, 2.0] -->
           </port>
           <port id="3">
               <dim>2</dim>  <!--The values in this input are [2, 3] (axes). -->
           </port>
       </input>
       <output>
           <port id="0"  precision="FP32">
               <dim>1</dim>
               <dim>2</dim>
               <dim>24</dim>
               <dim>160</dim>
           </port>
       </output>
   </layer>



