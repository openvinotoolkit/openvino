Interpolate
===========


.. meta::
  :description: Learn about I420toRGB-8 - an image processing operation, which
                can be performed on two required tensors.

**Versioned name**: *Interpolate-1*

**Category**: *Image processing*

**Short description**: *Interpolate* layer performs interpolation of independent slices in input tensor by specified dimensions and attributes.

**Attributes**

* *axes*

  * **Description**: ``axes`` specify spatial dimension indices where interpolation is applied. Other dimensions are treated as batch dimensions. The order of elements in ``axes`` attribute matters and mapped directly to elements with the same indices in the 2nd input ``target_spatial_shape``.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: ``int[]``
  * **Required**: *yes*

* *mode*

  * **Description**: specifies type of interpolation
  * **Range of values**: one of ``nearest``, ``linear``, ``cubic``, ``area``
  * **Type**: string
  * **Required**: *yes*

* *align_corners*

  * **Description**: *align_corners* is a flag that specifies whether to align corners or not. 1 means the alignment is applied, 0 means the alignment isn't applied.
  * **Range of values**: true or false
  * **Type**: ``boolean``
  * **Default value**: true
  * **Required**: *no*

* *antialias*

  * **Description**: *antialias* is a flag that specifies whether to perform anti-aliasing.
  * **Range of values**:
    * false - do not perform anti-aliasing
    * true - perform anti-aliasing
  * **Type**: boolean
  * **Default value**: false
  * **Required**: *no*

* *pads_begin*

  * **Description**: *pads_beg* specify the number of pixels to add to the beginning of the image being interpolated.
    This is a scalar that specifies padding for each spatial dimension.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: ``int``
  * **Default value**: 0
  * **Required**: *no*

* *pads_end*

  * **Description**: *pads_end* specify the number of pixels to add to the beginning of the image being interpolated.
    This is a scalar that specifies padding for each spatial dimension.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: ``int``
  * **Default value**: 0
  * **Required**: *no*

**Inputs**

*   **1**: ``data`` - Input tensor with data for interpolation. Type of elements is any supported floating-point type. **Required.**

*   **2**: ``target_spatial_shape`` - 1D tensor describing output shape for spatial axes. Number of elements matches the number of indices in *axes* attribute, the order matches as well. **Required.**

**Outputs**

* **1**: Resulting interpolated tensor with elements of the same type as input ``data`` tensor. The shape of the output matches input ``data`` shape except spatial dimensions mentioned in ``axes`` attribute. For other dimensions shape matches sizes from ``target_spatial_shape`` in order specified in ``axes``.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="Interpolate" ...>
       <data axes="2,3" align_corners="0" pads_begin="0" pads_end="0" mode="linear"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>2</dim>
               <dim>48</dim>
               <dim>80</dim>
           </port>
           <port id="1">
               <dim>2</dim>  <!--The values in this input are [50, 60] -->
           </port>
       </input>
       <output>
           <port id="0">
               <dim>1</dim>
               <dim>2</dim>
               <dim>50</dim>
               <dim>60</dim>
           </port>
       </output>
   </layer>

