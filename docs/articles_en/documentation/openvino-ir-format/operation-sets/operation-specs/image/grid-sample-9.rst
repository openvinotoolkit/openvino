GridSample
==========


.. meta::
  :description: Learn about GridSample-9 - an image processing operation, which
                can be performed on two required input tensors.

**Versioned name:** *GridSample-9*

**Category:** *Image processing*

**Short description:** *GridSample* performs interpolated sampling of pixels from the input image, using normalized, non-integer coordinates passed in as one of its inputs.

**Detailed description:** *GridSample* operates on a 4D input tensor representing an image. It calculates the output by selecting a location in the input image based on the values of the ``grid`` input. The latter contains a pair of float numbers for each output element that the operator is supposed to produce. Conceptually the operator behaves like *Gather* or *GatherND* but the difference is that the pixels to be selected are denoted by pairs of floats which belong to the range ``[-1, 1]``. Those values have to be denormalized first (mapped to the integer coordinates of the input tensor) and then the output value is calculated according to the interpolation ``mode``.

**Attributes**

* *align_corners*

  * **Description:** controls how the extrema values in the ``grid`` input tensor map to the border pixels of the input image. The value of -1 for both width and height can map either to the center of the border pixels or their left/top border. Similarly, the value of 1 can also map to the center of the border pixels or their right/bottom border. Inherently this means that the ``GridSample`` operation treats pixels as squares rather than infinitely small points.
  * **Range of values:**

    * ``false`` - map extrema values to the center of pixels
    * ``true`` - map extrema values to the borders of pixels

  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

* *mode*

  * **Description**: specifies the interpolation type used to calculate the output elements
  * **Range of values**: one of: ``bilinear``, ``bicubic`` or ``nearest``
  * **Type**: ``string``
  * **Default value**: bilinear
  * **Required**: *no*

* *padding_mode*

  * **Description**: controls the handling of out-of-bounds coordinates. The denormalized coordinates might fall outside of the input tensor's area(values outside the grid).
  * **Range of values**:

    * ``zeros`` - consider values in the padding to be zeros
    * ``border`` - the operator is supposed to select the nearest in-bounds pixel
    * ``reflection`` - repeatedly reflect the out-of bounds value until it points to a pixel that belongs to the image

  * **Type**: ``string``
  * **Default value**: zeros
  * **Required**: *no*

**Inputs**

* **1**: ``data`` - Input tensor of type ``T`` with data to be sampled. This input is expected to
  be a 4-dimensional tensor with NCHW layout. **Required.**
* **2**: ``grid`` - A 4-dimensional tensor containing normalized sampling coordinates(pairs of floats).
  The shape of this tensor is ``[N, H_out, W_out, 2]`` and the data type is ``T1``. **Required.**

**Outputs**

* **1**: A 4-dimensional tensor of type ``T`` with ``[N, C, H_out, W_out]`` shape.
  It contains the interpolated values calculated by this operator.

**Types**

* **T**: any type supported by OpenVINO.
* **T1**: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="GridSample" ...>
       <data align_corners="true" mode="nearest" padding_mode="border"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>3</dim>
               <dim>100</dim>
               <dim>100</dim>
           </port>
           <port id="1">
               <dim>1</dim>
               <dim>10</dim>
               <dim>10</dim>
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="0">
               <dim>1</dim>
               <dim>3</dim>
               <dim>10</dim>
               <dim>10</dim>
           </port>
       </output>
   </layer>


