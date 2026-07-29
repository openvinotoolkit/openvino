GroupConvolution
================


.. meta::
  :description: Learn about GroupConvolution-1 - a 1D, 2D or 3D, convolution operation, which
                can be performed on input and kernel tensors in OpenVINO.

**Versioned name**: *GroupConvolution-1*

**Category**: *Convolution*

**Short description**: Computes 1D, 2D or 3D GroupConvolution of input and kernel tensors.

**Detailed description**: Splits input into multiple groups, convolves them with group filters
as in regular convolution and concatenates the results. More thorough explanation can be found in
`ImageNet Classification with Deep Convolutional Neural Networks <https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`__

**Attributes**: The operation has the same attributes as a regular _Convolution_. Number of groups is derived from the kernel shape.

* *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the filter on the feature map over the ``(z, y, x)``
    axes for 3D convolutions and ``(y, x)`` axes for 2D convolutions. For example, *strides* equal ``4,2,1`` means sliding
    the filter 4 pixel at a time over depth dimension, 2 over height dimension and 1 over width dimension.
  * **Range of values**: positive integer numbers
  * **Type**: ``int[]``
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* is a number of pixels to add to the beginning along each axis. For example,
    *pads_begin* equal ``1,2`` means adding 1 pixel to the top of the input and 2 to the left of the input.
  * **Range of values**: positive integer numbers
  * **Type**: ``int[]``
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* is a number of pixels to add to the ending along each axis. For example,
    *pads_end* equal ``1,2`` means adding 1 pixel to the bottom of the input and 2 to the right of the input.
  * **Range of values**: positive integer numbers
  * **Type**: ``int[]``
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *dilations*

  * **Description**: *dilations* denotes the distance in width and height between elements (weights) in the filter.
    For example, *dilation* equal ``1,1`` means that all the elements in the filter are neighbors,
    so it is the same as for the usual convolution. *dilation* equal ``2,2`` means that all the elements in the
    filter are matched not to adjacent elements in the input matrix, but to those that are adjacent with distance 1.
  * **Range of values**: positive integer numbers
  * **Type**: ``int[]``
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* how the padding is calculated. Possible values:

    * *explicit* - use explicit padding values from *pads_begin* and *pads_end*.
    * *same_upper* - the input is padded to match the output size. In case of odd padding value an extra padding is added at the end.
    * *same_lower* - the input is padded to match the output size. In case of odd padding value an extra padding is added at the beginning.
    * *valid* - do not use padding.

  * **Type**: ``string``
  * **Default value**: explicit
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is specified.

**Inputs**:

* **1**: Input tensor of type *T* and rank 3, 4 or 5. Layout is ``[N, GROUPS * C_IN, Z, Y, X]``
  (number of batches, number of channels, spatial axes Z, Y, X). **Required.**
* **2**: Convolution kernel tensor of type *T* and rank 4, 5 or 6. Layout is ``[GROUPS, C_OUT, C_IN, Z, Y, X]``
  (number of groups, number of output channels, number of input channels, spatial axes Z, Y, X),

  * **Note** Number of groups is derived from the shape of the kernel and not specified by any attribute.
  * **Note**: Type of the convolution (1D, 2D or 3D) is derived from the rank of the input tensors and not specified by any attribute:

    * 1D convolution (input tensors rank 3) means that there is only one spatial axis X
    * 2D convolution (input tensors rank 4) means that there are two spatial axes Y, X
    * 3D convolution (input tensors rank 5) means that there are three spatial axes Z, Y, X

**Outputs**:

* **1**: Output tensor of type *T* and rank 3, 4 or 5. Layout is ``[N, GROUPS * C_OUT, Z, Y, X]``
  (number of batches, number of output channels, spatial axes Z, Y, X).

**Types**:

* *T*: any numeric type.

**Example**:

1D GroupConvolution

.. code-block:: xml
   :force:

   <layer type="GroupConvolution" ...>
       <data dilations="1" pads_begin="2" pads_end="2" strides="1" auto_pad="explicit"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>12</dim>
               <dim>224</dim>
           </port>
           <port id="1">
               <dim>4</dim>
               <dim>1</dim>
               <dim>3</dim>
               <dim>5</dim>
           </port>
       </input>
       <output>
           <port id="2" precision="FP32">
               <dim>1</dim>
               <dim>4</dim>
               <dim>224</dim>
           </port>
       </output>


2D GroupConvolution

.. code-block:: xml
   :force:

   <layer type="GroupConvolution" ...>
       <data dilations="1,1" pads_begin="2,2" pads_end="2,2" strides="1,1" auto_pad="explicit"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>12</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
           <port id="1">
               <dim>4</dim>
               <dim>1</dim>
               <dim>3</dim>
               <dim>5</dim>
               <dim>5</dim>
           </port>
       </input>
       <output>
           <port id="2" precision="FP32">
               <dim>1</dim>
               <dim>4</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
       </output>


3D GroupConvolution

.. code-block:: xml
   :force:

   <layer type="GroupConvolution" ...>
       <data dilations="1,1,1" pads_begin="2,2,2" pads_end="2,2,2" strides="1,1,1" auto_pad="explicit"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>12</dim>
               <dim>224</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
           <port id="1">
               <dim>4</dim>
               <dim>1</dim>
               <dim>3</dim>
               <dim>5</dim>
               <dim>5</dim>
               <dim>5</dim>
           </port>
       </input>
       <output>
           <port id="2" precision="FP32">
               <dim>1</dim>
               <dim>4</dim>
               <dim>224</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
       </output>



