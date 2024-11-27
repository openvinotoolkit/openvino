Col2Im
===================


.. meta::
  :description: Learn about Col2Im-15 - data movement operation which combines sliding blocks into an image tensor.

**Versioned name**: *Col2Im-15*

**Category**: *Data movement*

**Short description**: *Col2Im* operation constructs an image based on ``input`` tensor containing sliding data blocks (blocks of the image) and desired ``output_size``.

**Detailed description**

Consider an ``input`` tensor containing batches of image blocks of shape ``(N, C * Product(kernel_size), L)``, where:

* ``N`` is the batch dimension,
* ``C * Product(kernel_size)`` is the number of elements within a block (each block contains ``Product(kernel_size)`` vectors containing values from each channel ``C``),
* ``L`` is the total number of blocks calculated as follows:

L = product from d=1 to 2 of floor((output_size[d] + pads_begin[d] + pads_end[d] - dilation[d] * (kernel_size[d] - 1) - 1) / stride[d] + 1)

where ``d`` is over all spatial dimensions.

The ``input`` blocks are being moved into the ``output`` tensor of shape ``(N, C, output_size[0], output_size[1])`` by combining the values contained in blocks.

Non-batched inputs are also supported, in which case the ``input`` has shape ``(C * Product(kernel_size), L)`` and the output has shape ``(C, output_size[0], output_size[1])``.

**Attributes**:

* *strides*

  * **Description**: stride in the sliding blocks in the input spatial dimensions.
  * **Range of values**: 1D tensor of positive integer numbers
  * **Type**: *T_IDX*
  * **Default value**: [1, 1]
  * **Required**: *no*

* *dilations*

  * **Description**: controls local stride of the elements.
  * **Range of values**: 1D tensor of non-negative integer numbers
  * **Type**: *T_IDX*
  * **Default value**: [1, 1]
  * **Required**: *no*

* *pads_begin*

  * **Description**: *pads_begin* is a number of zero-value pixels to add to the beginning along each axis. For example, *pads_begin* equal "1,2" means adding 1 pixel to the top of the input and 2 to the left of the input.
  * **Range of values**: 1D tensor of non-negative integer numbers
  * **Type**: *T_IDX*
  * **Default value**: [0, 0]
  * **Required**: *no*

* *pads_end*

  * **Description**: *pads_end* is a number of zero-value pixels to add to the ending along each axis. For example, *pads_end* equal "1,2" means adding 1 pixel to the bottom of the input and 2 to the right of the input.
  * **Range of values**: 1D tensor of non-negative integer numbers
  * **Type**: *T_IDX*
  * **Default value**: [0, 0]
  * **Required**: *no*

**Inputs**

* **1**: *data*

  * **Description**: A batched 3D tensor of type *T* and shape ``(N, C * Product(kernel_size), L)`` or an unbatched 2D tensor of type *T* and shape ``(C * Product(kernel_size), L)``. **Required.**
  * **Range of values**: 1D tensor of non-negative integer numbers
  * **Type**: *T*

* **2**: *output_size*

  * **Description**: controls the shape of the spatial dimensions of the output image. **Required.**
  * **Range of values**: 1D tensor of two positive integer numbers (height and width)
  * **Type**: *T_IDX*

* **3**: *kernel_size*

  * **Description**: size of the sliding blocks. **Required.**
  * **Range of values**: 1D tensor of non-negative integer numbers
  * **Type**: *T_IDX*

**Outputs**

* **1**: The output tensor the output image of type *T* and shape:

  * ``(N, C, output_size[0], output_size[1])`` in case of batched input,
  * ``(C, output_size[0], output_size[1])`` in case of non-batched input.

**Types**

* *T*: any supported data type.
* *T_IDX*: ``int64`` or ``int32``.

**Examples**

All examples assume ``C = 3``.

*Example 1: default optional Parameters*

For inputs ``output_size`` = [16, 16] and ``kernel_size`` = [2, 2]

.. code-block:: xml
   :force:

    <layer ... type="Col2Im" ... >
        <input>
            <port id="0" precision="I32">
                <dim>3</dim>     <!-- batch_axis -->
                <dim>12</dim>    <!-- C * Product(kernel_size) -->
                <dim>225</dim>   <!-- L -->
            </port>
            <port id="1" precision="I32">
                <dim>2</dim>     <!-- output_size -->
            </port>
            <port id="2" precision="I32">
                <dim>2</dim>     <!-- kernel_size -->
            </port>
        </input>
        <output>
            <port id="1" precision="I32">
                <dim>3</dim>     <!-- batch_axis -->
                <dim>3</dim>     <!-- C -->
                <dim>16</dim>    <!-- output_size[0] -->
                <dim>16</dim>    <!-- output_size[1] -->
            </port>
        </output>
    </layer>


*Example 2: non-default dilations, padding and strides*

For inputs ``output_size`` = [16, 16] and ``kernel_size`` = [3, 3]

.. code-block:: xml
   :force:

    <layer ... type="Col2Im" ... >
        <data dilations="2,2" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
        <input>
            <port id="0" precision="I32">
                <dim>1</dim>     <!-- batch_axis -->
                <dim>27/dim>     <!-- C * Product(kernel_size) -->
                <dim>25</dim>    <!-- L -->
            </port>
            <port id="1" precision="I32">
                <dim>2</dim>     <!-- output_size -->
            </port>
            <port id="2" precision="I32">
                <dim>2</dim>     <!-- kernel_size -->
            </port>
        </input>
        <output>
            <port id="1" precision="I32">
                <dim>1</dim>     <!-- batch_axis -->
                <dim>3</dim>     <!-- C -->
                <dim>16</dim>    <!-- output_size[0] -->
                <dim>16</dim>    <!-- output_size[1] -->
            </port>
        </output>
    </layer>

*Example 3: non-default dilations and padding*

For inputs ``output_size`` = [32, 32] and ``kernel_size`` = [2, 2]

.. code-block:: xml
   :force:

    <layer ... type="Col2Im" ... >
        <data dilations="2,2" pads_begin="3,3" pads_end="3,3"/>
        <input>
            <port id="0" precision="I32">
                <dim>12</dim>    <!-- batch_axis -->
                <dim>12/dim>     <!-- C * Product(kernel_size) -->
                <dim>324</dim>   <!-- L -->
            </port>
            <port id="1" precision="I32">
                <dim>2</dim>     <!-- output_size -->
            </port>
            <port id="2" precision="I32">
                <dim>2</dim>     <!-- kernel_size -->
            </port>
        </input>
        <output>
            <port id="1" precision="I32">
                <dim>12</dim>    <!-- batch_axis -->
                <dim>3</dim>     <!-- C -->
                <dim>32</dim>    <!-- output_size[0] -->
                <dim>32</dim>    <!-- output_size[1] -->
            </port>
        </output>
    </layer>

*Example 4: default optional Parameters, unbatched*

For inputs ``output_size`` = [16, 16] and ``kernel_size`` = [2, 2]

.. code-block:: xml
   :force:

    <layer ... type="Col2Im" ... >
        <input>
            <port id="0" precision="I32">
                <dim>12</dim>    <!-- C * Product(kernel_size) -->
                <dim>225</dim>   <!-- L -->
            </port>
            <port id="1" precision="I32">
                <dim>2</dim>     <!-- output_size -->
            </port>
            <port id="2" precision="I32">
                <dim>2</dim>     <!-- kernel_size -->
            </port>
        </input>
        <output>
            <port id="1" precision="I32">
                <dim>3</dim>     <!-- C -->
                <dim>16</dim>    <!-- output_size[0] -->
                <dim>16</dim>    <!-- output_size[1] -->
            </port>
        </output>
    </layer>
