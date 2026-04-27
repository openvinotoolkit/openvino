BGRtoNV12
=========


.. meta::
  :description: Learn about BGRtoNV12-17 - an image processing operation, which
                can be performed to convert an image from BGR to NV12 format.

**Versioned name**: *BGRtoNV12-17*

**Category**: *Image processing*

**Short description**: *BGRtoNV12* performs image conversion from BGR to NV12 format.

**Detailed description**:

Similar to *RGBtoNV12* but the input channels for each pixel are reversed so that the first input channel is ``blue``, the second one is ``green``, the last one is ``red``. The same YUV conversion formulas are applied after remapping the channel order. See detailed conversion formulas in the :doc:`RGBtoNV12 description <rgb-to-nv12-17>`.

**Attributes:**

Same as specified for :doc:`RGBtoNV12 <rgb-to-nv12-17>` operation.

**Inputs:**

Input BGR image tensor shall have ``NHWC (also known as NYXC)`` layout. The height ``H`` and width ``W`` dimensions must be even numbers.

* **1**: Tensor of type *T*. **Required.** Dimensions:

  * ``N`` - batch dimension
  * ``H`` - height dimension of the image (must be even)
  * ``W`` - width dimension of the image (must be even)
  * ``C`` - channels dimension is equal to 3. The first channel is Blue, the second one is Green, the last one is Red

**Outputs:**

Same layout as specified for :doc:`RGBtoNV12 <rgb-to-nv12-17>` operation (single-plane or two-plane NV12 depending on the *single_plane* attribute).

**Types:**

* *T*: ``uint8`` or any supported floating-point type.


**Examples:**

*Example 1: single-plane output*

.. code-block:: xml
   :force:

    <layer ... type="BGRtoNV12">
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

    <layer ... type="BGRtoNV12">
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


