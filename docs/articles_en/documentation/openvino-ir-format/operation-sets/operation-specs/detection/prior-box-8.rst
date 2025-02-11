PriorBox
========


.. meta::
  :description: Learn about PriorBox-8 - an object detection operation,
                which can be performed on two required input tensors.

**Versioned name**: *PriorBox-8*

**Category**: *Object detection*

**Short description**: *PriorBox* operation generates prior boxes of specified sizes and aspect ratios across all dimensions.

**Detailed description**:

*PriorBox* computes coordinates of prior boxes by the following rules:

1.  First, it calculates *center_x* and *center_y* of a prior box:

.. math::

  W \equiv Width \quad Of \quad Image \\ H \equiv Height \quad Of \quad Image

*   If step equals 0:

.. math::

  center_x=(w+0.5) \\ center_y=(h+0.5)

*   else:

.. math::

  center_x=(w+offset)*step \\ center_y=(h+offset)*step \\ w \subset \left( 0, W \right ) \\ h \subset \left( 0, H \right )

2.  Then, it calculates coordinates of prior boxes for each :math:`s \subset \left( 0, min\_sizes \right )` :

.. math::

  xmin = \frac{\frac{center_x - s}{2}}{W}



 .. math::

  ymin = \frac{\frac{center_y - s}{2}}{H}


.. math::

  xmax = \frac{\frac{center_x + s}{2}}{W}


.. math::

  ymin = \frac{\frac{center_y + s}{2}}{H}

3. If *clip* attribute is set to true, each output value is clipped between :math:`\left< 0, 1 \right>`.

**Attributes**:

* *min_size (max_size)*

  * **Description**: *min_size (max_size)* is the minimum (maximum) box size in pixels.
  * **Range of values**: positive floating-point numbers
  * **Type**: ``float[]``
  * **Default value**: []
  * **Required**: *no*

* *aspect_ratio*

  * **Description**: *aspect_ratio* is a variance of aspect ratios. Duplicate values are ignored.
  * **Range of values**: a set of positive integer numbers
  * **Type**: ``float[]``
  * **Default value**: []
  * **Required**: *no*

* *flip*

  * **Description**: *flip* is a flag that denotes that each *aspect_ratio* is duplicated and flipped. For example, *flip* equals 1 and *aspect_ratio* equals ``[4.0,2.0]``, meaning that the aspect_ratio is equal to ``[4.0,2.0,0.25,0.5]``.
  * **Range of values**:

    * false or 0 - each *aspect_ratio* is flipped
    * true or 1  - each *aspect_ratio* is not flipped
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

* *clip*

  * **Description**: *clip* is a flag that denotes if each value in the output tensor should be clipped to the ``[0,1]`` interval.
  * **Range of values**:

    * false or 0 - clipping is not performed
    * true or 1 - each value in the output tensor is clipped to the ``[0,1]`` interval.
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

* *step*

  * **Description**: *step* is a distance between box centers.
  * **Range of values**: floating-point non-negative number
  * **Type**: `float`
  * **Default value**: 0
  * **Required**: *no*

* *offset*

  * **Description**: *offset* is a shift of a box to the top left corner respectively.
  * **Range of values**: floating-point non-negative number
  * **Type**: `float`
  * **Required**: *yes*

* *variance*

  * **Description**: *variance* denotes a variance of adjusting bounding boxes. The attribute could contain 0, 1, or 4 elements.
  * **Range of values**: floating-point positive numbers
  * **Type**: `float[]`
  * **Default value**: []
  * **Required**: *no*

* *scale_all_sizes*

  * **Description**: *scale_all_sizes* is a flag that denotes type of inference. For example, *scale_all_sizes* equals 0 means that *max_size* attribute is ignored.
  * **Range of values**:

    * false - *max_size* is ignored
    * true  - *max_size* is used
  * **Type**: `boolean`
  * **Default value**: true
  * **Required**: *no*

* *fixed_ratio*

  * **Description**: *fixed_ratio* is an aspect ratio of a box.
  * **Range of values**: a list of positive floating-point numbers
  * **Type**: ``float[]``
  * **Default value**: []
  * **Required**: *no*

* *fixed_size*

  * **Description**: *fixed_size* is an initial box size in pixels.
  * **Range of values**: a list of positive floating-point numbers
  * **Type**: ``float[]``
  * **Default value**: []
  * **Required**: *no*

* *density*

  * **Description**: *density* is the square root of the number of boxes of each type.
  * **Range of values**: a list of positive floating-point numbers
  * **Type**: ``float[]``
  * **Default value**: []
  * **Required**: *no*

* *min_max_aspect_ratios_order*

  * **Description**: *min_max_aspect_ratios_order* is a flag that denotes the order of output prior box. If set true, the output prior box is in [min, max, aspect_ratios] order, which is consistent with Caffe. Note that the order affects the weights order of the preceding convolution layer and does not affect the final detection results.
  * **Range of values**:

    * false - the output prior box is in [min, aspect_ratios, max] order
    * true  - the output prior box is in [min, max, aspect_ratios] order
  * **Type**: ``boolean``
  * **Default value**: true
  * **Required**: *no*

**Inputs**:

*   **1**: ``output_size`` - 1D tensor of type *T_INT* with two elements ``[height, width]``. Specifies the spatial size of generated grid with boxes. **Required.**

*   **2**: ``image_size`` - 1D tensor of type *T_INT* with two elements ``[image_height, image_width]``. Specifies shape of the image for which boxes are generated. **Required.**

**Outputs**:

*   **1**: 2D tensor of shape ``[2, 4 * height * width * priors_per_point]`` and type *T_OUT* with box coordinates. The ``priors_per_point`` is the number of boxes generated per each grid element. The number depends on operation attribute values.

**Types**

* *T_INT*: any supported integer type.
* *T_OUT*: supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer type="PriorBox" ...>
       <data aspect_ratio="2.0" clip="false" density="" fixed_ratio="" fixed_size="" flip="true" max_size="38.46" min_size="16.0" offset="0.5" step="16.0" variance="0.1,0.1,0.2,0.2"/>
       <input>
           <port id="0">
               <dim>2</dim>        <!-- values: [24, 42] -->
           </port>
           <port id="1">
               <dim>2</dim>        <!-- values: [384, 672] -->
           </port>
       </input>
       <output>
           <port id="2">
               <dim>2</dim>
               <dim>16128</dim>
           </port>
       </output>
   </layer>


