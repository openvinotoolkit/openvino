MaxPool
=======


.. meta::
  :description: Learn about MaxPool-1 - a pooling operation, which can
                be performed on a 3D, 4D or 5D input tensor.

**Versioned name**: *MaxPool-1*

**Category**: *Pooling*

**Short description**: Performs max pooling operation on input.

**Detailed description**: Input shape can be either 3D, 4D or 5D. Max Pooling operation is performed with the respect to input shape from the third dimension to the last dimension. If paddings are used then during the pooling calculation their value are ``-inf``. The Max Pooling operation involves sliding a filter over each channel of feature map and downsampling by choosing the biggest value within the region covered by the filter. `Article about max pooling in Convolutional Networks <https://deeplizard.com/learn/video/ZjM_XQa5s6s>`__.

**Attributes**: *Pooling* attributes are specified in the ``data`` node, which is a child of the layer node.

* *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the window on the feature map over the (z, y, x) axes for 3D poolings and (y, x) axes for 2D poolings. For example, *strides* equal "4,2,1" means sliding the window 4 pixel at a time over depth dimension, 2 over height dimension and 1 over width dimension.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* is a number of pixels to add to the beginning along each axis. For example, *pads_begin* equal "1,2" means adding 1 pixel to the top of the input and 2 to the left of the input.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* is a number of pixels to add to the ending along each axis. For example, *pads_end* equal "1,2" means adding 1 pixel to the bottom of the input and 2 to the right of the input.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *kernel*

  * **Description**: *kernel* is a size of each filter. For example, *kernel* equal (2, 3) means that each filter has height equal to 2 and width equal to 3.
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

    * *explicit*: use explicit padding values from ``pads_begin`` and ``pads_end``.
    * *same_upper (same_lower)* the input is padded to match the output size. In case of odd padding value an extra padding is added at the end (at the beginning).
    * *valid* - do not use padding.

  * **Type**: string
  * **Default value**: *explicit*
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is not equal to explicit.

**Inputs**:

* **1**: 3D, 4D or 5D input tensor of type *T*. **Required.**

**Outputs**:

* **1**: Input shape can be either ``[N, C, H]``, ``[N, C, H, W]`` or ``[N, C, H, W, D]``. Then the corresponding output shape will be ``[N, C, H_out]``, ``[N, C, H_out, W_out]`` or ``[N, C, H_out, W_out, D_out]``. Output tensor has the same data type as input tensor.

**Types**

* *T*: floating-point or integer type.

**Mathematical Formulation**

Output shape calculation based on ``auto_pad`` and ``rounding_type``:

* ``auto_pad = explicit`` and ``rounding_type = floor``
        ``H_out = floor(H + pads_begin[0] + pads_end[0] - kernel[0] / strides[0]) + 1``
        ``W_out = floor(W + pads_begin[1] + pads_end[1] - kernel[1] / strides[1]) + 1``
        ``D_out = floor(D + pads_begin[2] + pads_end[2] - kernel[2] / strides[2]) + 1``

* ``auto_pad = valid`` and ``rounding_type = floor``
      ``H_out = floor(H - kernel[0] / strides[0]) + 1``
      ``W_out = floor(W - kernel[1] / strides[1]) + 1``
      ``D_out = floor(D - kernel[2] / strides[2]) + 1``

* ``auto_pad = same_upper/same_lower`` and ``rounding_type = floor``
      ``H_out = H``
      ``W_out = W``
      ``D_out = D``

* ``auto_pad = explicit`` and ``rounding_type = ceil``
      ``H_out = ceil(H + pads_begin[0] + pads_end[0] - kernel[0] / strides[0]) + 1``
      ``W_out = ceil(W + pads_begin[1] + pads_end[1] - kernel[1] / strides[1]) + 1``
      ``D_out = ceil(D + pads_begin[2] + pads_end[2] - kernel[2] / strides[2]) + 1``

* ``auto_pad = valid`` and ``rounding_type = ceil``
      ``H_out = ceil(H - kernel[0] / strides[0]) + 1``
      ``W_out = ceil(W - kernel[1] / strides[1]) + 1``
      ``D_out = ceil(D - kernel[2] / strides[2]) + 1``

* ``auto_pad = same_upper/same_lower`` and ``rounding_type = ceil``
      ``H_out = H``
      ``W_out = W``
      ``D_out = D``

If ``H + pads_begin[i] + pads_end[i] - kernel[i]`` is not divided by ``strides[i]`` evenly then the result is rounded with the respect to ``rounding_type`` attribute.

1. Example 1 shows how *MaxPool* operates with 4D input using 2D kernel and ``auto_pad = explicit``

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
      output = [[[[-1, 2, 3, 3],
                  [4, 5, 5, -6],
                  [4, 8, 9, 9],
                  [-7, 8, 9, 9]]]]


2. Example 2 shows how *MaxPool* operates with 3D input using 1D kernel and ``auto_pad = valid``

   .. code-block:: sh

      input = [[[-1, 2, 3, 5, -7, 9, 1]]]
      strides = [1]
      kernel = [3]
      rounding_type = "floor"
      auto_pad = "valid"
      output = [[[3, 5, 5, 9, 9]]]


3. Example 3 shows how *MaxPool* operates with 4D input using 2D kernel and ``auto_pad = same_lower``

   .. code-block:: sh

      input = [[[[-1, 2, 3],
               [4, 5, -6],
               [-7, 8, 9]]]]
      strides = [1, 1]
      kernel = [2, 2]
      rounding_type = "floor"
      auto_pad = "same_lower"
      output = [[[[-1, 2, 3],
                  [4, 5, 5]
                  [4, 8, 9]]]]


4. Example 4 shows how *MaxPool* operates with 4D input using 2D kernel and ``auto_pad = same_upper``

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
      output = [[[[5, 5, 3],
                  [8, 9, 9]
                  [8, 9, 9]],
                 [[6, 5, 5],
                  [8, 2, 1],
                  [8, 2, -3]]]]


5. Example 5 shows how *MaxPool* operates with 4D input using 2D kernel, ``auto_pad = valid`` and ``rounding_type = ceil``

   .. code-block:: sh

      input = [[[[-1, 2, 3],
                 [4, 5, -6],
                 [-7, 8, 9]]]]
      strides = [2, 2]
      kernel = [2, 2]
      rounding_type = "ceil"
      auto_pad = "valid"
      output = [[[[5, 3],
                  [8, 9]]]]


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
       </output>
   </layer>



