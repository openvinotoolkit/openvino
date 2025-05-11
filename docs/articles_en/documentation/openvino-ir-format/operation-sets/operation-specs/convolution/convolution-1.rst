Convolution
===========


.. meta::
  :description: Learn about Convolution-1 - a 1D, 2D or 3D convolution operation, which
                can be performed on input and kernel tensors in OpenVINO.

**Versioned name**: *Convolution-1*

**Category**: *Convolution*

**Short description**: Computes 1D, 2D or 3D convolution (cross-correlation to be precise) of input and kernel tensors.

**Detailed description**: Basic building block of convolution is a dot product of input patch and kernel. Whole operation consist of multiple such computations over multiple input patches and kernels. More thorough explanation can be found in `Convolutional Neural Networks <http://cs231n.github.io/convolutional-networks/#conv>`__ and `Convolution operation <https://medium.com/apache-mxnet/convolutions-explained-with-ms-excel-465d6649831c>`__ .

For the convolutional layer, the number of output features in each dimension is calculated using the formula:

.. math::

   n_{out} = \left ( \frac{n_{in} + 2p - k}{s} \right ) + 1

The receptive field in each layer is calculated using the formulas:

* Jump in the output feature map:

  .. math::

     j_{out} = j_{in} \cdot s

* Size of the receptive field of output feature:

  .. math::

     r_{out} = r_{in} + ( k - 1 ) \cdot j_{in}

* Center position of the receptive field of the first output feature:

  .. math::

     start_{out} = start_{in} + ( \frac{k - 1}{2} - p ) \cdot j_{in}

* Output is calculated using the following formula:

  .. math::

     out = \sum_{i = 0}^{n}w_{i}x_{i} + b

**Attributes**:

* *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the filter on the feature map over the ``(z, y, x)`` axes for 3D convolutions and ``(y, x)`` axes for 2D convolutions. For example, *strides* equal ``4,2,1`` means sliding the filter 4 pixel at a time over depth dimension, 2 over height dimension and 1 over width dimension.
  * **Range of values**: integer values starting from 0
  * **Type**: ``int[]``
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* is a number of pixels to add to the beginning along each axis. For example, *pads_begin* equal ``1,2`` means adding 1 pixel to the top of the input and 2 to the left of the input.
  * **Range of values**: integer values starting from 0
  * **Type**: ``int[]``
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* is a number of pixels to add to the ending along each axis. For example, *pads_end* equal ``1,2`` means adding 1 pixel to the bottom of the input and 2 to the right of the input.
  * **Range of values**: integer values starting from 0
  * **Type**: ``int[]``
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *dilations*

  * **Description**: *dilations* denotes the distance in width and height between elements (weights) in the filter. For example, *dilation* equal ``1,1`` means that all the elements in the filter are neighbors, so it is the same as for the usual convolution. *dilation* equal ``2,2`` means that all the elements in the filter are matched not to adjacent elements in the input matrix, but to those that are adjacent with distance 1.
  * **Range of values**: integer value starting from 0
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

* **1**: Input tensor of type *T* and rank 3, 4 or 5. Layout is ``[N, C_IN, Z, Y, X]`` (number of batches, number of channels, spatial axes Z, Y, X). **Required.**
* **2**: Kernel tensor of type *T* and rank 3, 4 or 5. Layout is ``[C_OUT, C_IN, Z, Y, X]`` (number of output channels, number of input channels, spatial axes Z, Y, X). **Required.**
* **Note**: Type of the convolution (1D, 2D or 3D) is derived from the rank of the input tensors and not specified by any attribute:

  * 1D convolution (input tensors rank 3) means that there is only one spatial axis X
  * 2D convolution (input tensors rank 4) means that there are two spatial axes Y, X
  * 3D convolution (input tensors rank 5) means that there are three spatial axes Z, Y, X

**Outputs**:

* **1**: Output tensor of type *T* and rank 3, 4 or 5. Layout is ``[N, C_OUT, Z, Y, X]`` (number of batches, number of kernel output channels, spatial axes Z, Y, X).

**Types**:

* *T*: any numeric type.

**Example**:

1D Convolution

.. code-block:: xml
   :force:

   <layer type="Convolution" ...>
       <data dilations="1" pads_begin="0" pads_end="0" strides="2" auto_pad="valid"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>5</dim>
               <dim>128</dim>
           </port>
           <port id="1">
               <dim>16</dim>
               <dim>5</dim>
               <dim>4</dim>
           </port>
       </input>
       <output>
           <port id="2" precision="FP32">
               <dim>1</dim>
               <dim>16</dim>
               <dim>63</dim>
           </port>
       </output>
   </layer>


2D Convolution

.. code-block:: xml
   :force:

   <layer type="Convolution" ...>
       <data dilations="1,1" pads_begin="2,2" pads_end="2,2" strides="1,1" auto_pad="explicit"/>
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

3D Convolution

.. code-block:: xml
   :force:

   <layer type="Convolution" ...>
       <data dilations="2,2,2" pads_begin="0,0,0" pads_end="0,0,0" strides="3,3,3" auto_pad="explicit"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>7</dim>
               <dim>320</dim>
               <dim>320</dim>
               <dim>320</dim>
           </port>
           <port id="1">
               <dim>32</dim>
               <dim>7</dim>
               <dim>3</dim>
               <dim>3</dim>
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="2" precision="FP32">
               <dim>1</dim>
               <dim>32</dim>
               <dim>106</dim>
               <dim>106</dim>
               <dim>106</dim>
           </port>
       </output>
   </layer>



