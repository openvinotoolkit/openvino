MaxPool
=======


.. meta::
  :description: Learn about MaxPool-14 - a pooling operation, which can
                be performed on a 3D, 4D or 5D input tensor.

**Versioned name**: *MaxPool-14*

**Category**: *Pooling*

**Short description**: Performs the max pooling operation on input.

**Detailed description**: Input shape can be either 3D, 4D, or 5D. The max pooling operation is performed with respect to input shape from the third dimension to the last dimension. If paddings are used, during the pooling calculation their values are ``-inf``. The max pooling operation involves sliding a filter over each channel of a feature map and downsampling by choosing the largest value within the region covered by the filter.

**Attributes**: *Pooling* attributes are specified in the ``data`` node, which is a child of the layer node.

* *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the window on the feature map over the (z, y, x) axes for 3D poolings and (y, x) axes for 2D poolings. For example, *strides* equal to "4,2,1" means sliding the window 4 pixels at a time over depth dimension, 2 over height dimension, and 1 over width dimension.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Required**: *yes*

* *dilations*

  * **Description**: *dilations* specify the index of the next pixel to select when pooling. If not present, the dilation defaults to 1, meaning the adjacent pixel is chosen. A value of 2 indicates that one pixel is skipped and every other pixel is considered. Dilations specify one value for each spatial axis of the kernel: ``(z, y, x)`` for 3D poolings and ``(y, x)``  for 2D poolings.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Default value**: ``[1, 1, ...]``
  * **Required**: *no*

* *pads_begin*

  * **Description**: *pads_begin* is a number of pixels to add to the beginning along each axis. For example, *pads_begin* equal to "1,2" means adding 1 pixel to the top of the input and 2 to the left of the input. All added padding values are equal to negative infinity.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* is a number of pixels to add to the ending along each axis. For example, *pads_end* equal to "1,2" means adding 1 pixel to the bottom of the input and 2 to the right of the input. All added padding values are equal to negative infinity.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when the *auto_pad* attribute is specified.

* *kernel*

  * **Description**: *kernel* is a size of each filter. For example, *kernel* equal to (2, 3) means that each filter has height equal to 2 and width equal to 3.
  * **Range of values**: integer values starting from 1
  * **Type**: int[]
  * **Required**: *yes*

* *rounding_type*

  * **Description**: *rounding_type* is a type of rounding to be used to compute output shape. *ceil_torch* does not allow the last pooling to start in the padding area.
  * **Range of values**:
    * *floor*
    * *ceil*
    * *ceil_torch*
  * **Type**: string
  * **Default value**: *floor*
  * **Required**: *no*

* *auto_pad*

  * **Description**: *auto_pad* how the padding is calculated. Possible values:

    * *explicit*: explicit padding values from ``pads_begin`` and ``pads_end`` are used.
    * *same_upper (same_lower)* the input is padded to match the output size. In case of odd padding value, an extra padding is added at the end (at the beginning).
    * *valid* padding is not used.

  * **Type**: string
  * **Default value**: *explicit*
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is not equal to explicit.

* *index_element_type*

  * **Description**: the type of output tensor with indices
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i64"
  * **Required**: *No*

* *axis*

  * **Description**: indicator of the first dimension in the input shape that should be used to calculate the upper bound of allowed index output values. The upper bound is the product of dimensions starting from the one pointed by the 'axis' attribute until the end of the input shape.
  * **Range of values**: integer number. Negative value means counting dimension from the end. The range is ``[-R, R - 1]``, where ``R`` is the rank of the input tensor.
  * **Type**: int
  * **Default value**: 0
  * **Required**: *No*

**Inputs**:

* **1**: 3D, 4D, or 5D input tensor of type T. Required.

**Outputs**:

  * **1**: Input shape can be either ``[N, C, H]``, ``[N, C, H, W]``, or ``[N, C, H, W, D]``. The corresponding output shape is ``[N, C, H_out]``, ``[N, C, H_out, W_out]`` or ``[N, C, H_out, W_out, D_out]``. Output tensor has the same data type as the input tensor. Output shape calculation rules and examples can be found in :doc:`Pooling Operators shape inference rules <pooling_shape_rules>`.

  * **2**: Output tensor of type *T_IND* with indices of values selected by the pooling operation.
    Shape of this output matches the first output. The type of this output can be specified using the ``index_element_type`` attribute.
    Values are computed as indices in a tensor flattened to 1D, not considering padding. Examples for a 5D input tensor:

    * When ``axis == 0``, the values are in the range ``[0, N * C * H * W * D)``.
    * When ``axis == 2``, the values are in the range ``[0, H * W * D)``.

    .. note::

       The values of this output can only be calculated correctly if ``pads_value`` is set to ``-infinity``.


**Types**

* *T*: floating point or integer type.

* *T_IND*: ``int64`` or ``int32``.


**Examples**

.. code-block:: xml
   :force:

   <layer ... type="MaxPool" ... >
       <data auto_pad="same_upper" kernel="2,2" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
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
           <port id="2">
               <dim>1</dim>
               <dim>3</dim>
               <dim>32</dim>
               <dim>32</dim>
           </port>
       </output>
   </layer>

   <layer ... type="MaxPool" ... >
       <data auto_pad="explicit" kernel="2,2" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
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
               <dim>17</dim>
               <dim>17</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>3</dim>
               <dim>17</dim>
               <dim>17</dim>
           </port>
       </output>
   </layer>

   <layer ... type="MaxPool" ... >
       <data auto_pad="valid" kernel="2,2" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
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
               <dim>16</dim>
               <dim>16</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>3</dim>
               <dim>16</dim>
               <dim>16</dim>
           </port>
       </output>
   </layer>
