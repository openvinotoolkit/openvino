GroupConvolutionBackpropData
============================



.. meta::
  :description: Learn about GroupConvolutionBackpropData-1 - a 1D, 2D or 3D convolution operation, which
                can be performed on input and kernel tensors in OpenVINO.

**Versioned name**: *GroupConvolutionBackpropData-1*

**Category**: *Convolution*

**Short description**: Computes 1D, 2D or 3D *GroupConvolutionBackpropData* of input and kernel tensors.

**Detailed description**: Splits input and filters into multiple groups, computes *ConvolutionBackpropData*
on them and concatenates the results. It is equivalent to GroupConvolution and Convolution relationship.

**Attributes**: The operation has the same attributes as a *ConvolutionBackpropData*. Number of groups
is derived from the kernel shape.


* *strides*

  * **Description**: *strides* has the same definition as *strides* for a regular Convolution but applied in
    the backward way, for the output tensor.
  * **Range of values**: positive integers
  * **Type**: ``int[]``
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* has the same definition as *pads_begin* for a regular Convolution but applied in
    the backward way, for the output tensor. May be omitted, in which case pads are calculated automatically.
  * **Range of values**: non-negative integers
  * **Type**: ``int[]``
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* has the same definition as *pads_end* for a regular Convolution but applied
    in the backward way, for the output tensor. May be omitted, in which case pads are calculated automatically.
  * **Range of values**: non-negative integers
  * **Type**: ``int[]``
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *dilations*

  * **Description**: *dilations* has the same definition as *dilations* for a regular Convolution but applied
    in the backward way, for the output tensor.
  * **Range of values**: positive integers
  * **Type**: ``int[]``
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* has the same definition as *auto_pad* for a regular Convolution but applied
    in the backward way, for the output tensor.

    * *explicit* - use explicit padding values from *pads_begin* and *pads_end*.
    * *same_upper* - the input is padded to match the output size. In case of odd padding value an extra padding is added at the end.
    * *same_lower* - the input is padded to match the output size. In case of odd padding value an extra padding is added at the beginning.
    * *valid* - do not use padding.

  * **Type**: ``string``
  * **Default value**: explicit
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is specified.

* *output_padding*

  * **Description**: *output_padding* adds additional amount of paddings per each spatial axis in the output tensor.
    It unlocks more elements in the output allowing them to be computed. Elements are added at the higher coordinate
    indices for the spatial dimensions. Number of elements in *output_padding* list matches the number of spatial
    dimensions in input and output tensors.
  * **Range of values**: non-negative integer values
  * **Type**: ``int[]``
  * **Default value**: all zeros
  * **Required**: *no*

**Inputs**:

* **1**: Input tensor of type ``T1`` and rank 3, 4 or 5. Layout is ``[N, GROUPS * C_IN, Z, Y, X]``
  (number of batches, number of channels, spatial axes Z, Y, X). **Required.**
* **2**: Kernel tensor of type ``T1`` and rank 4, 5 or 6. Layout is ``[GROUPS, C_IN, C_OUT, X, Y, Z]``
  (number of groups, number of input channels, number of output channels, spatial axes X, Y, Z). **Required.**

* **3**: Output shape tensor of type ``T2`` and rank 1. It specifies spatial shape of the output. **Optional.**
* **Note** Number of groups is derived from the shape of the kernel and not specified by any attribute.
* **Note**: Type of the convolution (1D, 2D or 3D) is derived from the rank of the input tensors and not specified by any attribute:

      * 1D convolution (input tensors rank 3) means that there is only one spatial axis X
      * 2D convolution (input tensors rank 4) means that there are two spatial axes Y, X
      * 3D convolution (input tensors rank 5) means that there are three spatial axes Z, Y, X

**Outputs**:

* **1**: Output tensor of type ``T1`` and rank 3, 4 or 5 (the same as input *1*). Layout is ``[N, GROUPS * C_OUT, Z, Y, X]``
  (number of batches, number of kernel output channels, spatial axes Z, Y, X).

**Types**:

* *T1*: any numeric type.
* *T2*: any integer type.

**Example**

1D GroupConvolutionBackpropData

.. code-block:: xml
   :force:

   <layer id="5" name="upsampling_node" type="GroupConvolutionBackpropData">
       <data dilations="1" pads_begin="1" pads_end="1" strides="2"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>20</dim>
               <dim>224</dim>
           </port>
           <port id="1">
               <dim>4</dim>
               <dim>5</dim>
               <dim>2</dim>
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="0" precision="FP32">
               <dim>1</dim>
               <dim>8</dim>
               <dim>447</dim>
           </port>
       </output>
   </layer>


2D GroupConvolutionBackpropData

.. code-block:: xml
   :force:

   <layer id="5" name="upsampling_node" type="GroupConvolutionBackpropData">
       <data dilations="1,1" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>20</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
           <port id="1">
               <dim>4</dim>
               <dim>5</dim>
               <dim>2</dim>
               <dim>3</dim>
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="0" precision="FP32">
               <dim>1</dim>
               <dim>8</dim>
               <dim>447</dim>
               <dim>447</dim>
           </port>
       </output>
   </layer>


3D GroupConvolutionBackpropData

.. code-block:: xml
   :force:

   <layer id="5" name="upsampling_node" type="GroupConvolutionBackpropData">
       <data dilations="1,1,1" pads_begin="1,1,1" pads_end="1,1,1" strides="2,2,2"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>20</dim>
               <dim>224</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
           <port id="1">
               <dim>4</dim>
               <dim>5</dim>
               <dim>2</dim>
               <dim>3</dim>
               <dim>3</dim>
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="0" precision="FP32">
               <dim>1</dim>
               <dim>8</dim>
               <dim>447</dim>
               <dim>447</dim>
               <dim>447</dim>
           </port>
       </output>
   </layer>



