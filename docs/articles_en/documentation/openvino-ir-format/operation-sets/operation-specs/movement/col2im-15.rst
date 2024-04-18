.. {#openvino_docs_ops_type_Col2Im_15}

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

.. math::

    L = \prod_{d=1}^{2} \left\lfloor \frac{({\text{{output\_size}}[d] + \text{{pads\_begin}}[d] + \text{{pads\_end}}[d] - \text{{dilation}}[d] \times (\text{{kernel\_size}}[d] - 1) - 1}}{{\text{{stride}}[d]}}) + 1\rfloor


where ``d`` is over all spatial dimensions.

The ``input`` blocks are being moved into the ``output`` tensor of shape ``(N, C, output_size[0], output_size[1])`` by combining the values contained in blocks.

**Attributes**:

* *output_size*

  * **Description**: controls the shape of the spatial dimensions of the output image.
  * **Range of values**: 1D tensor of non-negative integer numbers
  * **Type**: *T_IND*
  * **Required**: *yes*

* *kernel_size*

  * **Description**: size of the sliding blocks.
  * **Range of values**: 1D tensor of non-negative integer numbers
  * **Type**: *T_IND*
  * **Required**: *yes*

* *dilations*

  * **Description**: controls local stride of the elements.
  * **Range of values**: 1D tensor of non-negative integer numbers
  * **Type**: *T_IND*
  * **Default value**: [1, 1]
  * **Required**: *no*

* *pads_begin*

  * **Description**: *pads_begin* is a number of zero-value pixels to add to the beginning along each axis. For example, *pads_begin* equal "1,2" means adding 1 pixel to the top of the input and 2 to the left of the input.
  * **Range of values**: 1D tensor of non-negative integer numbers
  * **Type**: *T_IND*
  * **Default value**: [0, 0]
  * **Required**: *no*

* *pads_end*

  * **Description**: *pads_end* is a number of zero-value pixels to add to the ending along each axis. For example, *pads_end* equal "1,2" means adding 1 pixel to the bottom of the input and 2 to the right of the input.
  * **Range of values**: 1D tensor of non-negative integer numbers
  * **Type**: *T_IND*
  * **Default value**: [0, 0]
  * **Required**: *no*

* *stride*

  * **Description**: stride in the sliding blocks in the input spatial dimensions.
  * **Range of values**: 1D tensor of non-negative integer numbers
  * **Type**: *T_IND*
  * **Default value**: [1, 1]
  * **Required**: *no*

**Inputs**

* **1**: A 4D tensor of type *T* and shape ``(N, C * Product(kernel_size), L)``. **Required.**

**Outputs**

* **1**: The output tensor the output image of shape ``(N, C, output_size[0], output_size[1])`` and type *T*.

**Types**

* *T*: any supported data type.
* *T_IND*: ``int64`` or ``int32``.

**Examples**

All examples assume ``C = 3``.

*Example 1: default optional Parameters*

.. code-block:: xml
   :force:

    <layer ... type="Col2Im" ... >
        <data output_size="16,16" kernel_size="2,2"/>
        <input>
            <port id="0" precision="I32">
                <dim>3</dim>     <!-- batch_axis -->
                <dim>12</dim>    <!-- C * Product(kernel_size) -->
                <dim>225</dim>   <!-- L -->
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


*Example 2: non-default dilation, padding and stride*

.. code-block:: xml
   :force:

    <layer ... type="Col2Im" ... >
        <data output_size="16,16" kernel_size="3,3" dilation="2,2" pads_begin="1,1" pads_end="1,1" stride="2,2"/>
        <input>
            <port id="0" precision="I32">
                <dim>1</dim>     <!-- batch_axis -->
                <dim>27/dim>     <!-- C * Product(kernel_size) -->
                <dim>25</dim>    <!-- L -->
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

*Example 2: non-default dilation and padding*

.. code-block:: xml
   :force:

    <layer ... type="Col2Im" ... >
        <data output_size="32,32" kernel_size="2,2" dilation="2,2" pads_begin="3,3" pads_end="3,3"/>
        <input>
            <port id="0" precision="I32">
                <dim>12</dim>    <!-- batch_axis -->
                <dim>12/dim>     <!-- C * Product(kernel_size) -->
                <dim>1296</dim>  <!-- L -->
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
