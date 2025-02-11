BinaryConvolution
=================


.. meta::
  :description: Learn about BinaryConvolution-1 - a 2D convolution operation, which
                can be performed on binary input and binary kernel tensors in OpenVINO.

**Versioned name**: *BinaryConvolution-1*

**Category**: *Convolution*

**Short description**: Computes 2D convolution of binary input and binary kernel tensors.

**Detailed description**: The operation behaves as regular *Convolution* but uses specialized algorithm for computations on binary data. More thorough explanation can be found in `Understanding Binary Neural Networks <https://sushscience.wordpress.com/2017/10/01/understanding-binary-neural-networks/>`__ and `Bitwise Neural Networks <https://saige.sice.indiana.edu/wp-content/uploads/icml2015_mkim.pdf>`__.


Computation algorithm for mode *xnor-popcount*:

- ``X = XNOR(input_patch, filter)``, where XNOR is bitwise operation on two bit streams
- ``P = popcount(X)``, where popcount is the number of ``1`` bits in the ``X`` bit stream
- ``Output = 2 * P - B``, where ``B`` is the total number of bits in the ``P`` bit stream

**Attributes**:

* *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the filter on the feature map over the ``(y, x)`` axes for 2D convolutions. For example, *strides* equal ``2,1`` means sliding the filter 2 pixel at a time over height dimension and 1 over width dimension.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* is a number of pixels to add to the beginning along each axis. For example, *pads_begin* equal ``1,2`` means adding 1 pixel to the top of the input and 2 to the left of the input.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* is a number of pixels to add to the ending along each axis. For example, *pads_end* equal ``1,2`` means adding 1 pixel to the bottom of the input and 2 to the right of the input.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *dilations*

  * **Description**: *dilations* denotes the distance in width and height between elements (weights) in the filter. For example, *dilation* equal ``1,1`` means that all the elements in the filter are neighbors, so it is the same as for the usual convolution. *dilation* equal ``2,2`` means that all the elements in the filter are matched not to adjacent elements in the input matrix, but to those that are adjacent with distance 1.
  * **Range of values**: integer value starting from 0
  * **Type**: int[]
  * **Required**: *yes*

* *mode*

  * **Description**: *mode* defines how input tensor ``0/1`` values and weights ``0/1`` are interpreted as real numbers and how the result is computed.
  * **Range of values**:

    * *xnor-popcount*
  * **Type**: ``string``
  * **Required**: *yes*
  * **Note**: value ``0`` in inputs is interpreted as ``-1``, value ``1`` as ``1``

* *pad_value*

  * **Description**: *pad_value* is a floating-point value used to fill pad area.
  * **Range of values**: a floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* how the padding is calculated. Possible values:

    * *explicit* - use explicit padding values from *pads_begin* and *pads_end*.
    * *same_upper* - the input is padded to match the output size. In case of odd padding value an extra padding is added at the end.
    * *same_lower* - the input is padded to match the output size. In case of odd padding value an extra padding is added at the beginning.
    * *valid* - do not use padding.
  * **Type**: string
  * **Default value**: explicit
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is specified.

**Inputs**:

*   **1**: Input tensor of type *T1* and rank 4. Layout is ``[N, C_IN, Y, X]`` (number of batches, number of channels, spatial axes Y, X). **Required.**
*   **2**: Kernel tensor of type *T2* and rank 4. Layout is ``[C_OUT, C_IN, Y, X]`` (number of output channels, number of input channels, spatial axes Y, X). **Required.**
*   **Note**: Interpretation of tensor values is defined by *mode* attribute.

**Outputs**:

*   **1**: Output tensor of type *T3* and rank 4. Layout is ``[N, C_OUT, Y, X]`` (number of batches, number of kernel output channels, spatial axes Y, X).

**Types**:

* *T1*: any numeric type with values ``0`` or ``1``.
* *T2*: ``u1`` type with binary values ``0`` or ``1``.
* *T3*: *T1* type with full range of values.

**Example**:

2D Convolution

.. code-block:: xml
   :force:

   <layer type="BinaryConvolution" ...>
       <data dilations="1,1" pads_begin="2,2" pads_end="2,2" strides="1,1" mode="xnor-popcount" pad_value="0" auto_pad="explicit"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>3</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
           <port id="1">
               <dim>64</dim>
               <dim>3</dim>
               <dim>5</dim>
               <dim>5</dim>
           </port>
       </input>
       <output>
           <port id="2" precision="FP32">
               <dim>1</dim>
               <dim>64</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
       </output>
   </layer>


