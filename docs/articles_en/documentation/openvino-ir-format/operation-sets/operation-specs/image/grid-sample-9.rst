GridSample
==========


.. meta::
  :description: Learn about GridSample-9 - an image processing operation, which
                can be performed on two required input tensors.

**Versioned name:** *GridSample-9*

**Category:** *Image processing*

**Short description:** *GridSample* performs interpolated sampling of pixels from the input image, using normalized, non-integer coordinates passed in as one of its inputs.

**Detailed description:** *GridSample* operates on a 4D input tensor representing an image. It calculates the output by selecting a location in the input image based on the values of the ``grid`` input. The latter contains a pair of float numbers for each output element that the operator is supposed to produce. Conceptually the operator behaves like *Gather* or *GatherND* but the difference is that the pixels to be selected are denoted by pairs of floats which belong to the range ``[-1, 1]``. Those values have to be denormalized first (mapped to the integer coordinates of the input tensor) and then the output value is calculated according to the interpolation ``mode``.

*GridSample* also supports 5D (volumetric) input. In that case the ``data`` tensor has the ``[N, C, D, H, W]`` layout and the ``grid`` tensor has the shape ``[N, D_out, H_out, W_out, 3]``, where the last dimension stores ``(x, y, z)`` triplets (``x`` maps to ``W``, ``y`` to ``H`` and ``z`` to ``D``). The ``bilinear`` mode performs trilinear interpolation over the 8 surrounding voxels and ``nearest`` selects the closest voxel. The ``bicubic`` mode is only defined for 4D input and is rejected for 5D. The number of spatial dimensions of ``data`` and the last dimension of ``grid`` must be consistent (``2`` for 4D, ``3`` for 5D).

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
  be a 4-dimensional tensor with NCHW layout, or a 5-dimensional tensor with NCDHW layout for
  volumetric sampling. **Required.**
* **2**: ``grid`` - A tensor containing normalized sampling coordinates. For 4D ``data`` it is a
  4-dimensional tensor of shape ``[N, H_out, W_out, 2]`` holding ``(x, y)`` pairs; for 5D ``data`` it
  is a 5-dimensional tensor of shape ``[N, D_out, H_out, W_out, 3]`` holding ``(x, y, z)`` triplets.
  The data type is ``T1``. **Required.**

**Outputs**

* **1**: A tensor of type ``T`` with ``[N, C, H_out, W_out]`` shape for 4D input, or
  ``[N, C, D_out, H_out, W_out]`` for 5D input.
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


