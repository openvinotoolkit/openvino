Interpolate
===========


.. meta::
  :description: Learn about Interpolate-11 - an image processing operation, which
                can be performed on two required and one optional tensor.

**Versioned name**: *Interpolate-11*

**Category**: *Image processing*

**Short description**: *Interpolate* layer performs interpolation of independent slices of the input tensor by specified dimensions and attributes.

**Attributes**

* *mode*

  * **Description**: specifies type of interpolation
  * **Range of values**: one of ``nearest``, ``linear``, ``linear_onnx``, ``cubic``, ``bilinear_pillow``, ``bicubic_pillow``
  * **Type**: string
  * **Required**: *yes*
  * **Note**: Only 2D, 3D, 4D, 5D tensors with ``axes = {0, 1}``, ``axes = {0, 1, 2}``, ``axes = {2, 3}``,  ``axes = {2, 3, 4}`` respectively are supported for ``"mode" == "linear_onnx"``. In case of ``bilinear_pillow`` or ``bicubic_pillow`` only the spatial dimensions (H, W) can be specified in the ``axes`` tensor, for example in case of NHWC layout the axes should contain ``axes = {1, 2}``.

* *shape_calculation_mode*

  * **Description**: specifies how the data in the ``scales_or_sizes`` input should be interpreted when determining the operator's output shape.
  * **Range of values**: name of a shape calculation mode in string format:
    * ``sizes`` - the output shape is calculated as ``output_shape[axes[i]] = scales_or_sizes[i]`` for all ``i in range(0, len(axes))`` and ``output_shape[j] = input_shape[j] + pads_begin[j] + pads_end[j]`` for ``j not in axes``, ``j in range(0, rank(image))``.
    * ``scales`` - an output shape is calculated as ``output_shape[axes[i]] = floor(scales_or_sizes[i] * (input_shape[axes[i]] + pads_begin[axes[i]] + pads_end[axes[i]]))`` for all ``i in range(0, len(axes))`` and ``output_shape[j] = input_shape[j] + pads_begin[j] + pads_end[j]`` for ``j not in axes``, ``j in range(0, rank(image))``
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
  * **Note**: When the selected interpolation mode is ``BILINEAR_PILLOW`` or ``BICUBIC_PILLOW`` this attribute is ignored.

* *nearest_mode*

  * **Description**: specifies the rounding mode when ``mode == nearest`` and is used only when ``mode == nearest``.
  * **Range of values**: name of the rounding mode in string format:
    * ``round_prefer_floor`` - this mode is known as round half down.
    * ``round_prefer_ceil`` - it is round half up mode.
    * ``floor`` - this mode computes the largest integer value not greater than the rounded value.
    * ``ceil`` - this mode computes the smallest integer value not less than the rounded value.
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
  * **Note**: When the selected interpolation mode is ``BILINEAR_PILLOW`` or ``BICUBIC_PILLOW`` this attribute is ignored. Pillow-kind of antialiasing is applied in those modes.

* *pads_begin*

  * **Description**: *pads_begin* specifies the number of pixels to add to the beginning of the image being interpolated. This addition of pixels is done before the interpolation calculation.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: ``int[]``
  * **Default value**: ``[0]``
  * **Required**: *no*

* *pads_end*

  * **Description**: *pads_end* specifies the number of pixels to add to the end of the image being interpolated. This addition of pixels is done before the interpolation calculation.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: ``int[]``
  * **Default value**: ``[0]``
  * **Required**: *no*

* *cube_coeff*

  * **Description**: *cube_coeff* specifies the parameter *a* for cubic interpolation (see, e.g.  `article <https://ieeexplore.ieee.org/document/1163711/)>`__.  *cube_coeff* is used only when ``mode == cubic`` or ``mode == bicubic_pillow``.
  * **Range of values**: floating-point number
  * **Type**: any of supported floating-point type
  * **Default value**: ``-0.75`` (applicable for ``mode == cubic``). The value compatible with ``BICUBIC_PILLOW`` needs to be manually set to ``-0.5``
  * **Required**: *no*

**Inputs**

*   **1**: ``image`` - tensor of type *T* with data for interpolation. **Required.**

*   **2**: ``scales_or_sizes`` - 1D tensor containing the data used to calculate the spatial output shape. The number of elements must match the number of values in the ``axes`` input tensor, the order needs to match as well. The type of this input tensor is either *T_SCALES* or *T_SIZES* depending on the value of the ``shape_calculation_mode`` attribute. **Required.**

*   **3**: ``axes`` - 1D tensor of type *T_AXES* specifying dimension indices where interpolation is applied, and ``axes`` is any unordered list of indices of different dimensions of input tensor, e.g. ``[0, 4]``, ``[4, 0]``, ``[4, 2, 1]``, ``[1, 2, 3]``. These indices should be non-negative integers from ``0`` to ``rank(image) - 1`` inclusively.  Input tensor's dimensions not specified in the ``axes`` tensor are not modified by the operator. The order of elements in ``axes`` attribute matters and is mapped directly to the elements in the 2nd input ``scales_or_sizes``. **Optional** with default value ``[0,1,...,rank(image) - 1]``. If the ``axes`` input is not provided the number of elements in the ``scales_or_sizes`` tensor needs to match the number of automatically generated axes.

**Outputs**

*   **1**: Resulting interpolated tensor with elements of the same type as input ``image`` tensor. The shape of the output matches input ``image`` shape except spatial dimensions mentioned in ``axes`` attribute. For other dimensions shape matches sizes from ``sizes`` in order specified in ``axes``.

**Types**

* *T*: any supported numeric type.
* *T_SIZES*: any supported integer type.
* *T_SCALES*: any supported floating-point type.
* *T_AXES*: any supported integer type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="Interpolate" ...>
       <data shape_calculation_mode="scales" pads_begin="0" pads_end="0" mode="bicubic_pillow"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>2</dim>
               <dim>48</dim>
               <dim>80</dim>
           </port>
           <port id="1">
               <dim>2</dim> <!--The values in this input are [24, 160] -->
           </port>
           <port id="2">
               <dim>2</dim> <!--The values in this input are [0.5, 2.0] -->
           </port>
           <port id="3">
               <dim>2</dim> <!--The values in this input are [2, 3] (axes). -->
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

