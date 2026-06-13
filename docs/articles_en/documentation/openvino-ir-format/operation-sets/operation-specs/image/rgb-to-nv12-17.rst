RGBtoNV12
=========


.. meta::
  :description: Learn about RGBtoNV12-17 - an image processing operation, which
                can be performed to convert an image from RGB to NV12 format.

**Versioned name**: *RGBtoNV12-17*

**Category**: *Image processing*

**Short description**: *RGBtoNV12* performs image conversion from RGB to NV12 format.

**Detailed description:**

Conversion of each pixel from RGB to NV12 (YUV) space is represented by the following formulas:

.. math::

   \begin{aligned} & Y = 0.257 \cdot R + 0.504 \cdot G + 0.098 \cdot B + 16 \\ & U = -0.148 \cdot R - 0.291 \cdot G + 0.439 \cdot B + 128 \\ & V = 0.439 \cdot R - 0.368 \cdot G - 0.071 \cdot B + 128 \end{aligned}


Then Y, U, V values are clipped to range (0, 255).

**Attributes:**

* *single_plane*

  * **Description**: Controls whether the NV12 output is packed into a single tensor or split into separate Y and UV planes.
  * **Range of values**: ``true`` or ``false``
  * **Type**: ``bool``
  * **Default value**: ``true``
  * **Required**: *no*

**Inputs:**

Input RGB image tensor shall have ``NHWC (also known as NYXC)`` layout. The height ``H`` and width ``W`` dimensions must be even numbers. Dimensions:

* **1**: Tensor of type *T*. **Required.** Dimensions:

  * ``N`` - batch dimension
  * ``H`` - height dimension of the image (must be even)
  * ``W`` - width dimension of the image (must be even)
  * ``C`` - channels dimension is equal to 3 (Red, Green, Blue)

**Outputs:**

Output NV12 image can be represented in two ways depending on the *single_plane* attribute:

* *Single plane* (``single_plane = true``):

  * **1**: Tensor of type *T*. Dimensions:

    * ``N`` - batch dimension
    * ``H`` - height dimension is 1.5x bigger than the image height (``image_height * 3 / 2``). The first ``image_height`` rows contain the Y (luma) plane; the remaining ``image_height / 2`` rows contain interleaved U and V chroma samples
    * ``W`` - width dimension is the same as the image width
    * ``C`` - channels dimension is equal to 1 (one plane)

* *Two separate planes - Y and UV* (``single_plane = false``):

  * **1**: Tensor of type *T* representing the Y (luma) plane. Dimensions:

    * ``N`` - batch dimension
    * ``H`` - height dimension is the same as the image height
    * ``W`` - width dimension is the same as the image width
    * ``C`` - channels dimension is equal to 1 (only Y channel)

  * **2**: Tensor of type *T* representing the UV (chroma) plane. Dimensions:

    * ``N`` - batch dimension. Shall be the same as the batch dimension for the Y plane
    * ``H`` - height dimension shall be half of the image height (for example, ``image_height / 2``)
    * ``W`` - width dimension shall be half of the image width (for example, ``image_width / 2``)
    * ``C`` - channels dimension shall be equal to 2 (interleaved U channel and V channel)

**Types:**

* *T*: ``uint8`` or any supported floating-point type.


**Examples:**

*Example 1: single-plane output*

.. code-block:: xml
   :force:

    <layer ... type="RGBtoNV12">
        <data single_plane="true"/>
        <input>
            <port id="0">
                <dim>1</dim>
                <dim>480</dim>
                <dim>640</dim>
                <dim>3</dim>
            </port>
        </input>
        <output>
            <port id="1">
                <dim>1</dim>
                <dim>720</dim>
                <dim>640</dim>
                <dim>1</dim>
            </port>
        </output>
    </layer>


*Example 2: two-plane output*

.. code-block:: xml
   :force:

    <layer ... type="RGBtoNV12">
        <data single_plane="false"/>
        <input>
            <port id="0">
                <dim>1</dim>
                <dim>480</dim>
                <dim>640</dim>
                <dim>3</dim>
            </port>
        </input>
        <output>
            <port id="1">  <!-- Y plane -->
                <dim>1</dim>
                <dim>480</dim>
                <dim>640</dim>
                <dim>1</dim>
            </port>
            <port id="2">  <!-- UV plane -->
                <dim>1</dim>
                <dim>240</dim>
                <dim>320</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>


