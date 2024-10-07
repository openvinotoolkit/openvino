DeformableConvolution
=====================


.. meta::
  :description: Learn about DeformableConvolution-1 - a 2D, deformable, convolution operation, which
                can be performed on input and kernel tensors in OpenVINO.

**Versioned name**: *DeformableConvolution-1*

**Category**: *Convolution*

**Short description**: Computes 2D deformable convolution of input and kernel tensors.

**Detailed description**: *Deformable Convolution* is similar to regular *Convolution* but its receptive field is deformed because of additional spatial offsets used during input sampling. More thorough explanation can be found in `Deformable Convolutions Demystified <https://towardsdatascience.com/deformable-convolutions-demystified-2a77498699e8>`__ and `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`__.

Output is calculated using the following formula:

.. math::

   y(p) = \displaystyle{\sum_{k = 1}^{K}}w_{k}x(p + p_{k} + {\Delta}p_{k})

Where

* K is a number of sampling locations, e.g. for kernel 3x3 and dilation = 1, K = 9
* :math:`x(p)` and :math:`y(p)` denote the features at location p from the input feature maps x and output feature maps y
* :math:`w_{k}` is the weight for k-th location.
* :math:`p_{k}` is pre-specified offset for the k-th location, e.g. K = 9 and :math:`p_{k} \in { (-1, -1),(-1, 0), . . . ,(1, 1) }`
* :math:`{\Delta}p_{k}` is the learnable offset for the k-th location.

**Attributes**:

* *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the filter on the feature map over the ``(y,x)`` axes. For example, *strides* equal ``2,1`` means sliding the filter 2 pixel at a time over height dimension and 1 over width dimension.
  * **Range of values**: integer values starting from ``0``
  * **Type**: ``int[]``
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* is a number of pixels to add to the beginning along each axis. For example, *pads_begin* equal ``1,2`` means adding 1 pixel to the top of the input and 2 to the left of the input.
  * **Range of values**: integer values starting from ``0``
  * **Type**: ``int[]``
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* is a number of pixels to add to the ending along each axis. For example, *pads_end* equal ``1,2`` means adding 1 pixel to the bottom of the input and 2 to the right of the input.
  * **Range of values**: integer values starting from ``0``
  * **Type**: ``int[]``
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *dilations*

  * **Description**: *dilations* denotes the distance in width and height between elements (weights) in the filter. For example, *dilation* equal ``1,1`` means that all the elements in the filter are neighbors, so it is the same as for the usual convolution. *dilation* equal ``2,2`` means that all the elements in the filter are matched not to adjacent elements in the input matrix, but to those that are adjacent with distance 1.
  * **Range of values**: integer value starting from ``0``
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


* *group*

  * **Description**: *group* is the number of groups which *output* and *input* should be split into. For example, *group* equal to 1 means that all filters are applied to the whole input (usual convolution), *group* equal to 2 means that both *input* and *output* channels are separated into two groups and the *i-th output* group is connected to the *i-th input* group channel. *group* equal to a number of output feature maps implies depth-wise separable convolution.
  * **Range of values**: integer value starting from ``1``
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*

* *deformable_group*

  * **Description**: *deformable_group* is the number of groups in which *offsets* input and *output* should be split into along the channel axis. Apply the deformable convolution using the i-th part of the offsets part on the i-th out.
  * **Range of values**: integer value starting from ``1``
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*

**Inputs**:

* **1**: Input tensor of type *T* and rank 4. Layout is ``NCYX`` (number of batches, number of channels, spatial axes Y and X). **Required.**
* **2**: Offsets tensor of type *T* and rank 4. Layout is ``NCYX`` (number of batches, *deformable_group* \* kernel_Y \* kernel_X \* 2, spatial axes Y and X). **Required.**
* **3**: Kernel tensor of type *T* and rank 4. Layout is ``OIYX`` (number of output channels, number of input channels, spatial axes Y and X). **Required.**


**Outputs**:

* **1**: Output tensor of type *T* and rank 4. Layout is ``NOYX`` (number of batches, number of kernel output channels, spatial axes Y and X).

**Types**:

* *T*: Any numeric type.

**Example**

2D DeformableConvolution (deformable_group=1)

.. code-block:: xml
   :force:

   <layer type="DeformableConvolution" ...>
       <data dilations="1,1" pads_begin="0,0" pads_end="0,0" strides="1,1" auto_pad="explicit"  group="1" deformable_group="1"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>4</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
           <port id="1">
               <dim>1</dim>
               <dim>50</dim>
               <dim>220</dim>
               <dim>220</dim>
           </port>
           <port id="2">
               <dim>64</dim>
               <dim>4</dim>
               <dim>5</dim>
               <dim>5</dim>
           </port>
       </input>
       <output>
           <port id="2" precision="FP32">
               <dim>1</dim>
               <dim>64</dim>
               <dim>220</dim>
               <dim>220</dim>
           </port>
       </output>
   </layer>

2D DeformableConvolution (deformable_group=4)

.. code-block:: xml
   :force:

   <layer type="DeformableConvolution" ...>
       <data dilations="1,1" pads_begin="0,0" pads_end="0,0" strides="1,1" auto_pad="explicit"  group="1" deformable_group="4"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>4</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
           <port id="1">
               <dim>1</dim>
               <dim>200</dim>
               <dim>220</dim>
               <dim>220</dim>
           </port>
           <port id="2">
               <dim>64</dim>
               <dim>4</dim>
               <dim>5</dim>
               <dim>5</dim>
           </port>
       </input>
       <output>
           <port id="2" precision="FP32">
               <dim>1</dim>
               <dim>64</dim>
               <dim>220</dim>
               <dim>220</dim>
           </port>
       </output>
   </layer>


