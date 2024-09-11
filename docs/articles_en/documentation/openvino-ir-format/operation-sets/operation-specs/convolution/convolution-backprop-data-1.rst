ConvolutionBackpropData
=======================


.. meta::
  :description: Learn about ConvolutionBackpropData-1 - a 1D, 2D or 3D convolution operation, which
                can be performed on input and kernel tensors in OpenVINO.

**Versioned name**: *ConvolutionBackpropData-1*

**Category**: *Convolution*

**Short description**: Computes 1D, 2D or 3D *ConvolutionBackpropData* operation with respect to the input and kernel tensors. Also known as a Transposed Convolution.

**Detailed description**:

ConvolutionBackpropData takes the input tensor, weights tensor and output shape and computes the output tensor of a given shape. The shape of the output can be specified as an input 1D integer tensor explicitly or determined by other attributes implicitly. If output shape is specified as an explicit input, shape of the output exactly matches the specified size and required amount of padding is computed. More thorough explanation can be found in `Transposed Convolutions <https://arxiv.org/abs/1603.07285>`__.

ConvolutionBackpropData accepts the same set of attributes as a regular Convolution operation and additionally ``output_padding`` attribute, but they are interpreted in a "backward way", so they are applied to the output of ConvolutionBackpropData, but not to the input. Refer to a regular :doc:`Convolution <convolution-1>` operation for detailed description of each Convolution attribute.

When output shape is specified as an input tensor ``output_shape`` then it specifies only spatial dimensions. No batch or channel dimension should be passed along with spatial dimensions. If ``output_shape`` is omitted, then ``pads_begin``, ``pads_end`` or ``auto_pad`` are used to determine output spatial shape ``[O_z, O_y, O_x]`` by input spatial shape ``[I_z, I_y, I_x]`` in the following way:

.. code-block:: xml
   :force:

   if auto_pads != None:
       pads_begin[i] = 0
       pads_end[i] = 0

   Y_i = stride[i] * (X_i - 1) + ((K_i - 1) * dilations[i] + 1) - pads_begin[i] - pads_end[i] + output_padding[i]

where ``K_i`` filter kernel dimension along spatial axis ``i``.

If ``output_shape`` is specified, ``pads_begin`` and ``pads_end`` are ignored, and ``auto_pad`` defines how to distribute padding amount around the tensor. In this case pads are determined based on the next formulas to correctly align input and output tensors:

.. code-block:: xml
   :force:

   total_padding[i] = stride[i] * (X_i - 1) + ((K_i - 1) * dilations[i] + 1) - output_shape[i] + output_padding[i]
   if auto_pads != SAME_UPPER:
       pads_begin[i] = total_padding[i] // 2
       pads_end[i] = total_padding[i] - pads_begin[i]
   else:
       pads_end[i] = total_padding[i] // 2
       pads_begin[i] = total_padding[i] - pads_end[i]

**Attributes**

* *strides*

  * **Description**: *strides* has the same definition as *strides* for a regular Convolution but applied in the backward way, for the output tensor.
  * **Range of values**: positive integers
  * **Type**: ``int[]``
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* has the same definition as *pads_begin* for a regular Convolution but applied in the backward way, for the output tensor. May be omitted specified, in which case pads are calculated automatically.
  * **Range of values**: non-negative integers
  * **Type**: ``int[]``
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* has the same definition as *pads_end* for a regular Convolution but applied in the backward way, for the output tensor. May be omitted, in which case pads are calculated automatically.
  * **Range of values**: non-negative integers
  * **Type**: ``int[]``
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *dilations*

  * **Description**: *dilations* has the same definition as *dilations* for a regular Convolution but applied in the backward way, for the output tensor.
  * **Range of values**: positive integers
  * **Type**: ``int[]``
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* has the same definition as *auto_pad* for a regular Convolution but applied in the backward way, for the output tensor.

    * *explicit*: use explicit padding values from ``pads_begin`` and ``pads_end``.
    * *same_upper* the input is padded to match the output size. In case of odd padding value an extra padding is added at the end.
    * *same_lower* the input is padded to match the output size. In case of odd padding value an extra padding is added at the beginning.
    * *valid* - do not use padding.
  * **Type**: ``string``
  * **Default value**: None
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is specified.

* *output_padding*

  * **Description**: *output_padding* adds additional amount of paddings per each spatial axis in the ``output`` tensor. It unlocks more elements in the output allowing them to be computed. Elements are added at the higher coordinate indices for the spatial dimensions. Number of elements in *output_padding* list matches the number of spatial dimensions in ``data`` and ``output`` tensors.
  * **Range of values**: non-negative integer values
  * **Type**: ``int[]``
  * **Default value**: all zeros
  * **Required**: *no*

**Inputs**:

* **1**: Input tensor of type *T1* and rank 3, 4 or 5. Layout is ``[N, C_INPUT, Z, Y, X]`` (number of batches, number of input channels, spatial axes Z, Y, X). **Required.**
* **2**: Convolution kernel tensor of type *T1* and rank 3, 4 or 5. Layout is ``[C_INPUT, C_OUTPUT, Z, Y, X]`` (number of input channels, number of output channels, spatial axes Z, Y, X). Spatial size of the kernel is derived from the shape of this input and aren't specified by any attribute. **Required.**
* **3**: ``output_shape`` is 1D tensor of type *T2* that specifies spatial shape of the output. If specified, *padding amount* is deduced from relation of input and output spatial shapes according to formulas in the description. If not specified, *output shape* is calculated based on the ``pads_begin`` and ``pads_end`` or completely according to ``auto_pad``. **Optional.**
* **Note**: Type of the convolution (1D, 2D or 3D) is derived from the rank of the input tensors and not specified by any attribute:

  * 1D convolution (input tensors rank 3) means that there is only one spatial axis X,
  * 2D convolution (input tensors rank 4) means that there are two spatial axes Y, X,
  * 3D convolution (input tensors rank 5) means that there are three spatial axes Z, Y, X.

**Outputs**:

*   **1**: Output tensor of type *T1* and rank 3, 4 or 5. Layout is ``[N, C_OUTPUT, Z, Y, X]`` (number of batches, number of kernel output channels, spatial axes Z, Y, X).

**Types**:

* *T1*: any numeric type.
* *T2*: any integer type.

**Examples**

*Example 1: 2D ConvolutionBackpropData*

.. code-block:: xml
   :force:

   <layer id="5" name="upsampling_node" type="ConvolutionBackpropData">
       <data dilations="1,1" pads_begin="1,1" pads_end="1,1" strides="2,2" output_padding="0,0" auto_pad="explicit"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>20</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
           <port id="1">
               <dim>20</dim>
               <dim>10</dim>
               <dim>3</dim>
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="0" precision="FP32">
               <dim>1</dim>
               <dim>10</dim>
               <dim>447</dim>
               <dim>447</dim>
           </port>
       </output>
   </layer>

*Example 2: 2D ConvolutionBackpropData with output_padding*

.. code-block:: xml
   :force:

   <layer id="5" name="upsampling_node" type="ConvolutionBackpropData">
       <data dilations="1,1" pads_begin="0,0" pads_end="0,0" strides="3,3" output_padding="2,2" auto_pad="explicit"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>20</dim>
               <dim>2</dim>
               <dim>2</dim>
           </port>
           <port id="1">
               <dim>20</dim>
               <dim>10</dim>
               <dim>3</dim>
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="0" precision="FP32">
               <dim>1</dim>
               <dim>10</dim>
               <dim>8</dim>
               <dim>8</dim>
           </port>
       </output>
   </layer>

*Example 3: 2D ConvolutionBackpropData with output_shape input*

.. code-block:: xml
   :force:

   <layer id="5" name="upsampling_node" type="ConvolutionBackpropData">
       <data dilations="1,1" pads_begin="1,1" pads_end="1,1" strides="1,1" output_padding="0,0" auto_pad="valid"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>20</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
           <port id="1">
               <dim>20</dim>
               <dim>10</dim>
               <dim>3</dim>
               <dim>3</dim>
           </port>
           <port id="2">
               <dim>2</dim> <!-- output_shape value is: [450, 450]-->
           </port>
       </input>
       <output>
           <port id="0" precision="FP32">
               <dim>1</dim>
               <dim>10</dim>
               <dim>450</dim>
               <dim>450</dim>
           </port>
       </output>
   </layer>


