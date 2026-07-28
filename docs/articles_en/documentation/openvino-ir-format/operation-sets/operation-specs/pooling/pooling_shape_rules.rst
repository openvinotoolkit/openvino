Shape Calculation Rules for Pooling Operators
=============================================

.. meta::
  :description: Learn about output shape calculation rules for OpenVINO Pooling Operators.

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
    Please note that AvgPool does not support ``dilations`` attribute, in wchich case its value should be replaced with ``1``.
      ``H_out = ceil((H - ((kernel[0] - 1) * dilations[0] + 1) + 1) / strides[0])``
      ``W_out = ceil((W - ((kernel[1] - 1) * dilations[1] + 1) + 1) / strides[1])``
      ``D_out = ceil((D - ((kernel[2] - 1) * dilations[2] + 1) + 1) / strides[2])``

* ``auto_pad = same_upper / same_lower``
      ``H_out = H``
      ``W_out = W``
      ``D_out = D``


If ``H + pads_begin[i] + pads_end[i] - kernel[i]`` is not divisible by ``strides[i]`` evenly, the result is rounded with respect to the ``rounding_type`` attribute.
If ``rounding_type`` is set to ``ceil_torch``, the last pooling operation within a dimension cannot start in the padding area. If this is the case, the respective dimension is reduced by ``1``. More context can be found in the `PyTorch issue discussion <https://github.com/pytorch/pytorch/issues/57178>`__.

**Examples**

1. Example 1 shows how *MaxPool* operates with 4D input using 2D kernel and ``auto_pad = explicit``.

   .. code-block:: sh

      input = [[[[-1, 2, 3],
                 [4, 5, -6],
                 [-7, 8, 9]]]]   # shape: (1, 1, 3, 3)
      strides = [1, 1]
      pads_begin = [1, 1]
      pads_end = [1, 1]
      kernel = [2, 2]
      rounding_type = "floor"
      auto_pad = "explicit"
      output0 = [[[[-1, 2, 3, 3],
                   [4, 5, 5, -6],
                   [4, 8, 9, 9],
                   [-7, 8, 9, 9]]]]   # shape: (1, 1, 4, 4)
      output1 = [[[[0, 1, 2, 2],
                   [3, 4, 4, 5],
                   [3, 7, 8, 8],
                   [6, 7, 8, 8]]]]   # shape: (1, 1, 4, 4)


2. Example 2 shows how *MaxPool* operates with 3D input using 1D kernel and ``auto_pad = valid``.

   .. code-block:: sh

      input = [[[-1, 2, 3, 5, -7, 9, 1]]]   # shape: (1, 1, 7)
      strides = [1]
      kernel = [3]
      rounding_type = "floor"
      auto_pad = "valid"
      output0 = [[[3, 5, 5, 9, 9]]]   # shape: (1, 1, 5)
      output1 = [[[2, 3, 3, 5, 5]]]   # shape: (1, 1, 5)


3. Example 3 shows how *MaxPool* operates with 4D input using 2D kernel and ``auto_pad = same_lower``.

   .. code-block:: sh

      input = [[[[-1, 2, 3],
               [4, 5, -6],
               [-7, 8, 9]]]]   # shape: (1, 1, 3, 3)
      strides = [1, 1]
      kernel = [2, 2]
      rounding_type = "floor"
      auto_pad = "same_lower"
      output0 = [[[[-1, 2, 3],
                  [4, 5, 5]
                  [4, 8, 9]]]]   # shape: (1, 1, 3, 3)
      output1 = [[[[0, 1, 2],
                  [3, 4, 4],
                  [3, 7, 8]]]]   # shape: (1, 1, 3, 3)


4. Example 4 shows how *MaxPool* operates with 4D input using 2D kernel and ``auto_pad = same_upper``.

   .. code-block:: sh

      input = [[[[-1, 2, 3],
                 [4, 5, -6],
                 [-7, 8, 9]],
                [[2, -1, 5],
                 [6, -7, 1],
                 [8, 2, -3]]]]   # shape: (1, 2, 3, 3)
      strides = [1, 1]
      kernel = [2, 2]
      rounding_type = "floor"
      auto_pad = "same_upper"
      output0 = [[[[5, 5, 3],
                   [8, 9, 9]
                   [8, 9, 9]],
                  [[6, 5, 5],
                   [8, 2, 1],
                   [8, 2, -3]]]]   # shape: (1, 2, 3, 3)
      output1 = [[[[4, 4, 2],
                   [7, 8, 8],
                   [7, 8, 8]],
                  [[12, 11, 11],
                   [15, 16, 14],
                   [15, 16, 17]]]]   # shape: (1, 2, 3, 3)


5. Example 5 shows how *MaxPool* operates with 4D input using 2D kernel and ``rounding_type = ceil_torch``.

   .. code-block:: sh

      input = [[[[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]]]   # shape: (1, 1, 3, 3)
      strides = [2, 2]
      kernel = [2, 2]
      pads_begin = [1, 1]
      pads_end = [1, 1]
      rounding_type = "ceil_torch"
      output0 = [[[[1, 3],
                   [7, 9]]]]   # shape: (1, 1, 2, 2)
      output1 = [[[[0, 2],
                   [6, 8]]]]   # shape: (1, 1, 2, 2)


6. Example 6 shows how *MaxPool* operates with 4D input using 2D kernel, ``auto_pad = valid`` and ``rounding_type = ceil``.

   .. code-block:: sh

      input = [[[[-1, 2, 3],
                 [4, 5, -6],
                 [-7, 8, 9]]]]   # shape: (1, 1, 3, 3)
      strides = [2, 2]
      kernel = [2, 2]
      rounding_type = "ceil"
      auto_pad = "valid"
      output0 = [[[[5, 3],
                   [8, 9]]]]   # shape: (1, 1, 2, 2)
      output1 = [[[[4, 2],
                   [7, 8]]]]   # shape: (1, 1, 2, 2)


7. Example 7 shows how *MaxPool* operates on 4D input using dilated 2D kernel, ``auto_pad = explicit`` and ``rounding_type = floor``.

   .. code-block:: sh

      input = [[[[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]]]   # shape: (1, 1, 3, 3)
      strides = [1, 1]
      kernel = [2, 2]
      dilations = [2, 2]
      rounding_type = "floor"
      auto_pad = "explicit"
      pads_begin = [1, 1]
      pads_end = [1, 1]
      output0 = [[[[5, 6, 5],
                   [8, 9, 8],
                   [5, 6, 5]]]]   # shape: (1, 1, 3, 3)
      output1 = [[[[4, 5, 4],
                   [7, 8, 7],
                   [4, 5, 4]]]]   # shape: (1, 1, 3, 3)


8. Example 8 shows how *MaxPool* operates on 4D input using 2D kernel, with non-default ``axis`` value.

Input shape:   (1, 2, 3, 3)
Output shape:  (1, 2, 2, 2)

   .. code-block:: sh

      input = [[[[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]],
                [[10, 11, 12],
                 [13, 14, 15],
                 [16, 17, 18]]]]   # shape: (1, 2, 3, 3)
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
                   [17, 18]]]]   # shape: (1, 2, 2, 2)
      output1 = [[[[4, 5],
                   [7, 8]],
                  [[4, 5],
                   [7, 8]]]]   # shape: (1, 2, 2, 2)
