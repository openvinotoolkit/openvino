AvgPool
=======


.. meta::
  :description: Learn about AvgPool-1 - a pooling operation, which can
                be performed on a 3D, 4D or 5D input tensor.

**Versioned name**: *AvgPool-1*

**Category**: *Pooling*

**Short description**: `Reference <http://caffe.berkeleyvision.org/tutorial/layers/pooling.html>`__

**Detailed description**: `Reference <http://cs231n.github.io/convolutional-networks/#pool>`__ . Average Pool is a pooling operation that performs down-sampling by dividing the input into pooling regions of size specified by kernel attribute and computing the average values of each region. Output shape is calculated as follows:

``H_out = (H + pads_begin[0] + pads_end[0] - kernel[0] / strides[0]) + 1``

``W_out = (H + pads_begin[1] + pads_end[1] - kernel[1] / strides[1]) + 1``

``D_out = (H + pads_begin[2] + pads_end[2] - kernel[2] / strides[2]) + 1``

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

* *exclude-pad*

  * **Description**: *exclude-pad* is a type of pooling strategy for values in the padding area. For example, if *exclude-pad* is "true", then zero-values that came from padding are not included in averaging calculation.
  * **Range of values**: true or false
  * **Type**: boolean
  * **Required**: *yes*

* *rounding_type*

  * **Description**: *rounding_type* is a type of rounding to be applied.
  * **Range of values**:

    * *ceil*
    * *floor*
  * **Type**: string
  * **Default value**: *floor*
  * **Required**: *no*

* *auto_pad*

  * **Description**: *auto_pad* how the padding is calculated. Possible values:

    * *explicit*: use explicit padding values from `pads_begin` and `pads_end`.
    * *same_upper (same_lower)* the input is padded to match the output size. In case of odd padding value an extra padding is added at the end (at the beginning).
    * *valid* - do not use padding.
  * **Type**: string
  * **Default value**: *explicit*
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is specified.

**Inputs**:

* **1**: 3D, 4D or 5D input tensor. **Required.**

**Outputs**:

* **1**: Input shape can be either ``[N,C,H]``, ``[N,C,H,W]`` or ``[N,C,H,W,D]``. Then the corresponding output shape is ``[N,C,H_out]``, ``[N,C,H_out,W_out]`` or ``[N,C,H_out,W_out,D_out]``.

**Mathematical Formulation**

.. math::

   output_{j} = \frac{\sum_{i = 0}^{n}x_{i}}{n}

**Examples**

.. code-block:: xml
   :force:

   <layer ... type="AvgPool" ... >
       <data auto_pad="same_upper" exclude-pad="true" kernel="2,2" pads_begin="0,0" pads_end="1,1" strides="2,2"/>
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

   <layer ... type="AvgPool" ... >
       <data auto_pad="same_upper" exclude-pad="false" kernel="5,5" pads_begin="0,0" pads_end="1,1" strides="2,2"/>
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

   <layer ... type="AvgPool" ... >
       <data auto_pad="explicit" exclude-pad="true" kernel="5,5" pads_begin="1,1" pads_end="1,1" strides="3,3"/>
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
               <dim>10</dim>
               <dim>10</dim>
           </port>
       </output>
   </layer>

   <layer ... type="AvgPool" ... >
       <data auto_pad="explicit" exclude-pad="false" kernel="5,5" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
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
               <dim>15</dim>
               <dim>15</dim>
           </port>
       </output>
   </layer>

   <layer ... type="AvgPool" ... >
       <data auto_pad="valid" exclude-pad="true" kernel="5,5" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
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
               <dim>14</dim>
               <dim>14</dim>
           </port>
       </output>
   </layer>


