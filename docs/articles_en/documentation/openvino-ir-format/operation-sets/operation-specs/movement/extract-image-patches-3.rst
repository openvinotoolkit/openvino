ExtractImagePatches
===================


.. meta::
  :description: Learn about ExtractImagePatches-3 - a data movement operation,
                which can be performed on a 4D input tensor.

**Versioned name**: *ExtractImagePatches-3*

**Category**: *Data movement*

**Short description**: The *ExtractImagePatches* operation collects patches from the input tensor, as if applying a convolution. All extracted patches are stacked in the depth dimension of the output.

**Detailed description**:

The *ExtractImagePatches* operation extracts patches of shape ``sizes`` which are ``strides`` apart in the input image. The output elements are taken from the input at intervals given by the ``rate`` argument, as in dilated convolutions.

The result is a 4D tensor containing image patches with size ``size[0] * size[1] * depth`` vectorized in the "depth" dimension.

The "auto_pad" attribute has no effect on the size of each patch, it determines how many patches are extracted.


**Attributes**

* *sizes*

  * **Description**: *sizes* is a size ``[size_rows, size_cols]`` of the extracted patches.
  * **Range of values**: non-negative integer number
  * **Type**: ``int[]``
  * **Required**: *yes*

* *strides*

  * **Description**: *strides* is a distance ``[stride_rows, stride_cols]`` between centers of two consecutive patches in an input tensor.
  * **Range of values**: non-negative integer number
  * **Type**: ``int[]``
  * **Required**: *yes*

* *rates*

  * **Description**: *rates* is the input stride ``[rate_rows, rate_cols]``, specifying how far two consecutive patch samples are in the input. Equivalent to extracting patches with ``patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1)``, followed by subsampling them spatially by a factor of rates. This is equivalent to rate in dilated (a.k.a. Atrous) convolutions.
  * **Range of values**: non-negative integer number
  * **Type**: ``int[]``
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* how the padding is calculated. Possible values:

    * *same_upper (same_lower)* the input is padded by zeros to match the output size. In case of odd padding value an extra padding is added at the end (at the beginning).
    * *valid* - do not use padding.
  * **Type**: ``string``
  * **Required**: *yes*

**Inputs**

* **1**: ``data`` the 4-D tensor of type *T* with shape ``[batch, depth, in_rows, in_cols]``. **Required.**

**Outputs**

* **1**: 4-D tensor with shape ``[batch, size[0] * size[1] * depth, out_rows, out_cols]`` with type equal to ``data`` tensor. Note ``out_rows`` and ``out_cols`` are the dimensions of the output patches.

**Types**

* *T*: any supported type.

**Example**

.. code-block:: xml
   :force:

   <layer type="ExtractImagePatches" ...>
       <data sizes="3,3" strides="5,5" rates="1,1" auto_pad="valid"/>
       <input>
           <port id="0">
               <dim>64</dim>
               <dim>3</dim>
               <dim>10</dim>
               <dim>10</dim>
           </port>
       </input>
       <output>
           <port id="1" precision="f32">
               <dim>64</dim>
               <dim>27</dim>
               <dim>2</dim>
               <dim>2</dim>
           </port>
       </output>
   </layer>

Image is a ``1 x 1 x 10 x 10`` array that contains the numbers 1 through 100. We use the symbol ``x`` to mark output patches.

1. ``sizes="3,3", strides="5,5", rates="1,1", auto_pad="valid"``

   .. math::

      \begin{bmatrix}
          x & x & x & 4 & 5 & x & x & x & 9 & 10 \\
          x & x & x & 14 & 15 & x & x & x & 19 & 20 \\
          x & x & x & 24 & 25 & x & x & x & 29 & 30 \\
          31 & 32 & 33 & 34 & 35 & 36 & 37 & 38 & 39 & 40 \\
          41 & 42 & 43 & 44 & 45 & 46 & 47 & 48 & 49 & 50 \\
          x & x & x & 54 & 55 & x & x & x & 59 & 60 \\
          x & x & x & 64 & 65 & x & x & x & 69 & 70 \\
          x & x & x & 74 & 75 & x & x & x & 79 & 80 \\
          81 & 82 & 83 & 84 & 85 & 86 & 87 & 88 & 89 & 90 \\
          91 & 92 & 93 & 94 & 95 & 96 & 79 & 98 & 99 & 100
      \end{bmatrix}


   output:

   .. code-block:: cpp

      [[[[ 1  6]
         [51 56]]

        [[ 2  7]
         [52 57]]

        [[ 3  8]
         [53 58]]

        [[11 16]
         [61 66]]

        [[12 17]
         [62 67]]

        [[13 18]
         [63 68]]

        [[21 26]
         [71 76]]

        [[22 27]
         [72 77]]

        [[23 28]
         [73 78]]]]

   output shape: `[1, 9, 2, 2]`

2. ``sizes="4,4", strides="8,8", rates="1,1", auto_pad="valid"``

    .. math::

      \begin{bmatrix}
          x & x & x & x & 5 & 6 & 7 & 8 & 9 & 10 \\
          x & x & x & x & 15 & 16 & 17 & 18 & 19 & 20 \\
          x & x & x & x & 25 & 26 & 27 & 28 & 29 & 30 \\
          x & x & x & x & 35 & 36 & 37 & 38 & 39 & 40 \\
          41 & 42 & 43 & 44 & 45 & 46 & 47 & 48 & 49 & 50 \\
          51 & 52 & 53 & 54 & 55 & 56 & 57 & 58 & 59 & 60 \\
          61 & 62 & 63 & 64 & 65 & 66 & 67 & 68 & 69 & 70 \\
          71 & 72 & 73 & 74 & 75 & 76 & 77 & 78 & 79 & 80 \\
          81 & 82 & 83 & 84 & 85 & 86 & 87 & 88 & 89 & 90 \\
          91 & 92 & 93 & 94 & 95 & 96 & 79 & 98 & 99 & 100
      \end{bmatrix}


    output:

    .. code-block:: cpp

       [[[[ 1]]

        [[ 2]]

        [[ 3]]

        [[ 4]]

        [[11]]

        [[12]]

        [[13]]

        [[14]]

        [[21]]

        [[22]]

        [[23]]

        [[24]]

        [[31]]

        [[32]]

        [[33]]

        [[34]]]]

    output shape: ``[1, 16, 1, 1]``

3. ``sizes="4,4", strides="9,9", rates="1,1", auto_pad="same_upper"``

   .. math::

      \begin{bmatrix}
          x & x & x & x & 0 & 0 & 0 & 0 & 0 & x & x & x & x\\
          x & x & x & x & 4 & 5 & 6 & 7 & 8 & x & x & x & x\\
          x & x & x & x & 14 & 15 & 16 & 17 & 18 & x & x & x & x\\
          x & x & x & x & 24 & 25 & 26 & 27 & 28 & x & x & x & x\\
          0 & 31 & 32 & 33 & 34 & 35 & 36 & 37 & 38 & 39 & 40 & 0 & 0\\
          0 & 41 & 42 & 43 & 44 & 45 & 46 & 47 & 48 & 49 & 50 & 0 & 0\\
          0 & 51 & 52 & 53 & 54 & 55 & 56 & 57 & 58 & 59 & 60 & 0 & 0\\
          0 & 61 & 62 & 63 & 64 & 65 & 66 & 67 & 68 & 69 & 70 & 0 & 0\\
          0 & 71 & 72 & 73 & 74 & 75 & 76 & 77 & 78 & 79 & 80 & 0 & 0\\
          x & x & x & x & 84 & 85 & 86 & 87 & 88 & x & x & x & x\\
          x & x & x & x & 94 & 95 & 96 & 79 & 98 & x & x & x & x\\
          x & x & x & x & 0 & 0 & 0 & 0 & 0 & x & x & x & x\\
          x & x & x & x & 0 & 0 & 0 & 0 & 0 & x & x & x & x
      \end{bmatrix}

   output:

   .. code-block:: cpp

      [[[[  0   0]
         [  0  89]]

        [[  0   0]
         [ 81  90]]

        [[  0   0]
         [ 82   0]]

        [[  0   0]
         [ 83   0]]

        [[  0   9]
         [  0  99]]

        [[  1  10]
         [ 91 100]]

        [[  2   0]
         [ 92   0]]

        [[  3   0]
         [ 93   0]]

        [[  0  19]
         [  0   0]]

        [[ 11  20]
         [  0   0]]

        [[ 12   0]
         [  0   0]]

        [[ 13   0]
         [  0   0]]

        [[  0  29]
         [  0   0]]

        [[ 21  30]
         [  0   0]]

        [[ 22   0]
         [  0   0]]

        [[ 23   0]
         [  0   0]]]]

   output shape: ``[1, 16, 2, 2]``

4. ``sizes="3,3", strides="5,5", rates="2,2", auto_pad="valid"``

   This time we use the symbols ``x``, ``y``, ``z`` and ``k`` to distinguish the patches:

   .. math::

      \begin{bmatrix}
          x & 2 & x & 4 & x & y & 7 & y & 9 & y \\
          11 & 12 & 13 & 14 & 15 & 16 & 17 & 18 & 19 & 20 \\
          x & 22 & x & 24 & x & y & 27 & y & 29 & y \\
          31 & 32 & 33 & 34 & 35 & 36 & 37 & 38 & 39 & 40 \\
          x & 42 & x & 44 & x & y & 47 & y & 49 & y \\
          z & 52 & z & 54 & z & k & 57 & k & 59 & k \\
          61 & 62 & 63 & 64 & 65 & 66 & 67 & 68 & 69 & 70 \\
          z & 72 & z & 74 & z & k & 77 & k & 79 & k \\
          81 & 82 & 83 & 84 & 85 & 86 & 87 & 88 & 89 & 90 \\
          z & 92 & z & 94 & z & k & 79 & k & 99 & k
      \end{bmatrix}

   output:

   .. code-block:: cpp

      [[[[  1   6]
         [ 51  56]]

        [[  3   8]
         [ 53  58]]

        [[  5  10]
         [ 55  60]]

        [[ 21  26]
         [ 71  76]]

        [[ 23  28]
         [ 73  78]]

        [[ 25  30]
         [ 75  80]]

        [[ 41  46]
         [ 91  96]]

        [[ 43  48]
         [ 93  98]]

        [[ 45  50]
         [ 95 100]]]]

   output_shape: ``[1, 9, 2, 2]``

5. ``sizes="2,2", strides="3,3", rates="1,1", auto_pad="valid"``

   Image is a ``1 x 2 x 5 x 5`` array that contains two feature maps where feature map with coordinate 0 contains numbers in a range ``[1, 25]`` and feature map with coordinate 1 contains numbers in a range ``[26, 50]``

   .. math::

      \begin{bmatrix}
          x & x & 3 & x & x\\
          x & x & 8 & x & x\\
          11 & 12 & 13 & 14 & 15\\
          x & x & 18 & x & x\\
          x & x & 23 & x & x
      \end{bmatrix}\\
      \begin{bmatrix}
          x & x & 28 & x & x\\
          x & x & 33 & x & x\\
          36 & 37 & 38 & 39 & 40\\
          x & x & 43 & x & x\\
          x & x & 48 & x & x
      \end{bmatrix}

   output:

   .. code-block:: cpp

      [[[[ 1  4]
         [16 19]]

        [[26 29]
         [41 44]]

        [[ 2  5]
         [17 20]]

        [[27 30]
         [42 45]]

        [[ 6  9]
         [21 24]]

        [[31 34]
         [46 49]]

        [[ 7 10]
         [22 25]]

        [[32 35]
         [47 50]]]]

   output shape: ``[1, 8, 2, 2]``


