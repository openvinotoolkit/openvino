MaxPool
=======


.. meta::
  :description: Learn about MaxPool-8 - a pooling operation, which can
                be performed on a 3D, 4D or 5D input tensor.

**Versioned name**: *MaxPool-8*

**Category**: *Pooling*

**Short description**: Performs the max pooling operation on input.

**Detailed description**: Input shape can be either 3D, 4D, or 5D. The max pooling operation is performed with respect to input shape from the third dimension to the last dimension. If paddings are used, during the pooling calculation their values are ``-inf``. The max pooling operation involves sliding a filter over each channel of a feature map and downsampling by choosing the largest value within the region covered by the filter.

**Attributes**: *Pooling* attributes are specified in the ``data`` node, which is a child of the layer node.

* *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the window on the feature map over the (z, y, x) axes for 3D poolings and (y, x) axes for 2D poolings. For example, *strides* equal to "4,2,1" means sliding the window 4 pixels at a time over depth dimension, 2 over height dimension, and 1 over width dimension.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Required**: *yes*

* *dilations*

  * **Description**: *dilations* specify the index of the next pixel to select when pooling. If not present, the dilation defaults to 1, meaning the adjacent pixel is chosen. A value of 2 indicates that one pixel is skipped and every other pixel is considered. Dilations specify one value for each spatial axis of the kernel: ``(z, y, x)`` for 3D poolings and ``(y, x)``  for 2D poolings.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Default value**: ``[1,1,...]``
  * **Required**: *no*

* *pads_begin*

  * **Description**: *pads_begin* is a number of pixels to add to the beginning along each axis. For example, *pads_begin* equal to "1,2" means adding 1 pixel to the top of the input and 2 to the left of the input. All added padding values are equal to negative infinity.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* is a number of pixels to add to the ending along each axis. For example, *pads_end* equal to "1,2" means adding 1 pixel to the bottom of the input and 2 to the right of the input. All added padding values are equal to negative infinity.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when the *auto_pad* attribute is specified.

* *kernel*

  * **Description**: *kernel* is a size of each filter. For example, *kernel* equal to (2, 3) means that each filter has height equal to 2 and width equal to 3.
  * **Range of values**: integer values starting from 1
  * **Type**: int[]
  * **Required**: *yes*

* *rounding_type*

  * **Description**: *rounding_type* is a type of rounding to be used to compute output shape.
  * **Range of values**:

    * *ceil*
    * *floor*

  * **Type**: string
  * **Default value**: *floor*
  * **Required**: *no*

* *auto_pad*

  * **Description**: *auto_pad* how the padding is calculated. Possible values:

    * *explicit*: explicit padding values from ``pads_begin`` and ``pads_end`` are used.
    * *same_upper (same_lower)* the input is padded to match the output size. In case of odd padding value, an extra padding is added at the end (at the beginning).
    * *valid* padding is not used.

  * **Type**: string
  * **Default value**: *explicit*
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is not equal to explicit.

* *index_element_type*

  * **Description**: the type of output tensor with indices
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i64"
  * **Required**: *No*

* *axis*

  * **Description**: indicator of the first dimension in the input shape that should be used to calculate the upper bound of allowed index output values. The upper bound is the product of dimensions starting from the one pointed by the 'axis' attribute until the end of the input shape.
  * **Range of values**: integer number. Negative value means counting dimension from the end. The range is ``[-R, R - 1]``, where ``R`` is the rank of the input tensor.
  * **Type**: int
  * **Default value**: 0
  * **Required**: *No*

**Inputs**:

* **1**: 3D, 4D, or 5D input tensor of type T. Required.

**Outputs**:

* **1**: Input shape can be either ``[N, C, H]``, ``[N, C, H, W]``, or ``[N, C, H, W, D]``. The corresponding output shape is ``[N, C, H_out]``, ``[N, C, H_out, W_out]`` or ``[N, C, H_out, W_out, D_out]``. Output tensor has the same data type as the input tensor.

* **2**: Output tensor of type *T_IND* with indices of values selected by the pooling operation.
  Shape of this output matches the first output. The type of this output can be specified using the ``index_element_type`` attribute.
  Values are computed as indices in a tensor flattened to 1D, not considering padding. Examples for a 5D input tensor:

  * When ``axis == 0``, the values are in the range ``[0, N * C * H * W * D)``.
  * When ``axis == 2``, the values are in the range ``[0, H * W * D)``.

  .. note::

      The values of this output can only be calculated correctly if ``pads_value`` is set to ``-infinity``.


**Types**

* *T*: floating point or integer type.

* *T_IND*: ``int64`` or ``int32``.


**Mathematical Formulation**

Output shape calculation based on ``auto_pad`` and ``rounding_type``:

* ``auto_pad = explicit`` and ``rounding_type = floor``
      ``H_out = floor((H + pads_begin[0] + pads_end[0] - ((kernel[0] - 1) * dilations[0] + 1)) / strides[0] + 1)``
      ``W_out = floor((W + pads_begin[1] + pads_end[1] - ((kernel[1] - 1) * dilations[1] + 1)) / strides[1] + 1)``
      ``D_out = floor((D + pads_begin[2] + pads_end[2] - ((kernel[2] - 1) * dilations[2] + 1)) / strides[2] + 1)``

* ``auto_pad = explicit`` and ``rounding_type = ceil``
      ``H_out = ceil((H + pads_begin[0] + pads_end[0] - ((kernel[0] - 1) * dilations[0] + 1)) / strides[0] + 1)``
      ``W_out = ceil((W + pads_begin[1] + pads_end[1] - ((kernel[1] - 1) * dilations[1] + 1)) / strides[1] + 1)``
      ``D_out = ceil((D + pads_begin[2] + pads_end[2] - ((kernel[2] - 1) * dilations[2] + 1)) / strides[2] + 1)``

* ``auto_pad = valid``
      ``H_out = ceil((H - ((kernel[0] - 1) * dilations[0] + 1) + 1) / strides[0])``
      ``W_out = ceil((W - ((kernel[1] - 1) * dilations[1] + 1) + 1) / strides[1])``
      ``D_out = ceil((D - ((kernel[2] - 1) * dilations[2] + 1) + 1) / strides[2])``

* ``auto_pad = same_upper / same_lower``
      ``H_out = H``
      ``W_out = W``
      ``D_out = D``


If ``H + pads_begin[i] + pads_end[i] - kernel[i]`` is not divisible by ``strides[i]`` evenly, the result is rounded with respect to the ``rounding_type`` attribute.

1. Example 1 shows how *MaxPool* operates with 4D input using 2D kernel and ``auto_pad = explicit``.

   .. code-block:: sh

      input = [[[[-1, 2, 3],
                 [4, 5, -6],
                 [-7, 8, 9]]]]
      strides = [1, 1]
      pads_begin = [1, 1]
      pads_end = [1, 1]
      kernel = [2, 2]
      rounding_type = "floor"
      auto_pad = "explicit"
      output0 = [[[[-1, 2, 3, 3],
                   [4, 5, 5, -6],
                   [4, 8, 9, 9],
                   [-7, 8, 9, 9]]]]
      output1 = [[[[0, 1, 2, 2],
                   [3, 4, 4, 5],
                   [3, 7, 8, 8],
                   [6, 7, 8, 8]]]]


2. Example 2 shows how *MaxPool* operates with 3D input using 1D kernel and ``auto_pad = valid``.

   .. code-block:: sh

      input = [[[-1, 2, 3, 5, -7, 9, 1]]]
      strides = [1]
      kernel = [3]
      rounding_type = "floor"
      auto_pad = "valid"
      output0 = [[[3, 5, 5, 9, 9]]]
      output1 = [[[2, 3, 3, 5, 5]]]


3. Example 3 shows how *MaxPool* operates with 4D input using 2D kernel and ``auto_pad = same_lower``.

   .. code-block:: sh

      input = [[[[-1, 2, 3],
               [4, 5, -6],
               [-7, 8, 9]]]]
      strides = [1, 1]
      kernel = [2, 2]
      rounding_type = "floor"
      auto_pad = "same_lower"
      output0 = [[[[-1, 2, 3],
                  [4, 5, 5]
                  [4, 8, 9]]]]
      output1 = [[[[0, 1, 2],
                  [3, 4, 4]
                  [3, 7, 8]]]]


4. Example 4 shows how *MaxPool* operates with 4D input using 2D kernel and ``auto_pad = same_upper``.


   .. code-block:: sh

      input = [[[[-1, 2, 3],
                 [4, 5, -6],
                 [-7, 8, 9]],
                [[2, -1, 5],
                 [6, -7, 1],
                 [8, 2, -3]]]]
      strides = [1, 1]
      kernel = [2, 2]
      rounding_type = "floor"
      auto_pad = "same_upper"
      output0 = [[[[5, 5, 3],
                   [8, 9, 9]
                   [8, 9, 9]],
                  [[6, 5, 5],
                   [8, 2, 1],
                   [8, 2, -3]]]]
      output1 = [[[[4, 4, 2],
                   [7, 8, 8]
                   [7, 8, 8]],
                  [[12, 11, 11],
                   [15, 16, 14],
                   [15, 16, 17]]]]


5. Example 5 shows how *MaxPool* operates with 4D input using 2D kernel, ``auto_pad = valid`` and ``rounding_type = ceil``.

   .. code-block:: sh

      input = [[[[-1, 2, 3],
                 [4, 5, -6],
                 [-7, 8, 9]]]]
      strides = [2, 2]
      kernel = [2, 2]
      rounding_type = "ceil"
      auto_pad = "valid"
      output0 = [[[[5, 3],
                   [8, 9]]]]
      output1 = [[[[4, 2],
                   [7, 8]]]]


6. Example 6 shows how *MaxPool* operates on 4D input using dilated 2D kernel, ``auto_pad = explicit`` and ``rounding_type = floor``.

   .. code-block:: sh

      input = [[[[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]]]
      strides = [1, 1]
      kernel = [2, 2]
      dilations = [2, 2]
      rounding_type = "floor"
      auto_pad = "explicit"
      pads_begin = [1, 1]
      pads_end = [1, 1]
      output0 = [[[[5, 6, 5],
                   [8, 9, 8],
                   [5, 6, 5]]]]
      output1 = [[[[4, 5, 4],
                   [7, 8, 7],
                   [4, 5, 4]]]]


7. Example 7 shows how *MaxPool* operates on 4D input using 2D kernel, with non-default ``axis`` value.

   .. code-block:: sh

      input = [[[[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]],
                [[10, 11, 12],
                 [13, 14, 15],
                 [16, 17, 18]]
                 ]]
      strides = [1, 1]
      kernel = [2, 2]
      dilations = [1, 1]
      rounding_type = "floor"
      auto_pad = "explicit"
      pads_begin = [0, 0]
      pads_end = [0, 0]
      axis = 2
      output0 = [[[[5, 6],
                   [8, 9]],
                  [[14, 15],
                   [17, 18]]]]
      output1 = [[[[4, 5],
                   [7, 8]],
                  [[4, 5],
                   [7, 8]]]]


**Examples**

.. code-block:: xml
   :force:

   <layer ... type="MaxPool" ... >
       <data auto_pad="same_upper" kernel="2,2" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>3</dim>
               <dim>32</dim>
               <dim>32</dim>
           </port>
       </input>
       <output>
           <port id="1">
               <dim>1</dim>
               <dim>3</dim>
               <dim>32</dim>
               <dim>32</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>3</dim>
               <dim>32</dim>
               <dim>32</dim>
           </port>
       </output>
   </layer>

   <layer ... type="MaxPool" ... >
       <data auto_pad="explicit" kernel="2,2" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>3</dim>
               <dim>32</dim>
               <dim>32</dim>
           </port>
       </input>
       <output>
           <port id="1">
               <dim>1</dim>
               <dim>3</dim>
               <dim>17</dim>
               <dim>17</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>3</dim>
               <dim>17</dim>
               <dim>17</dim>
           </port>
       </output>
   </layer>

   <layer ... type="MaxPool" ... >
       <data auto_pad="valid" kernel="2,2" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>3</dim>
               <dim>32</dim>
               <dim>32</dim>
           </port>
       </input>
       <output>
           <port id="1">
               <dim>1</dim>
               <dim>3</dim>
               <dim>16</dim>
               <dim>16</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>3</dim>
               <dim>16</dim>
               <dim>16</dim>
           </port>
       </output>
   </layer>



