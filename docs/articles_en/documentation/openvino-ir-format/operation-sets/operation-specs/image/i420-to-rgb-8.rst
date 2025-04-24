I420toRGB
=========


.. meta::
  :description: Learn about I420toRGB-8 - an image processing operation, which
                can be performed to convert image from I420 to RGB format.

**Versioned name**: *I420toRGB-8*

**Category**: *Image processing*

**Short description**: *I420toRGB* performs image conversion from I420 to RGB format.

**Detailed description:**

Conversion of each pixel from I420 (YUV) to RGB space is represented by the following formulas (same as in :doc:`NV12toRGB <nv12-to-rgb-8>`):

.. math::

   \begin{aligned}
   & R = 1.164 \cdot (Y - 16) + 1.596 \cdot (V - 128) \\
   & G = 1.164 \cdot (Y - 16) - 0.813 \cdot (V - 128) - 0.391 \cdot (U - 128) \\
   & B = 1.164 \cdot (Y - 16) + 2.018 \cdot (U - 128)
   \end{aligned}


Then ``R``, ``G``, ``B`` values are clipped to range ``(0, 255)``.

**Inputs:**

Input I420 image tensor shall have ``NHWC (also known as NYXC)`` layout and can be represented in two ways:

* *Single plane*:

  * **1**: Tensor of type *T*. **Required.** Dimensions:

    * ``N`` - batch dimension
    * ``H`` - height dimension is 1.5x bigger than the image height
    * ``W`` - width dimension is the same as the image width
    * ``C`` - channels dimension is equal to 1 (one plane)

* *Three separate planes - Y, U and V*:

  * **1**: Tensor of type *T* representing Y plane. **Required.** Dimensions:

    * ``N`` - batch dimension
    * ``H`` - height dimension is the same as the image height
    * ``W`` - width dimension is the same as the image width
    * ``C`` - channels dimension is equal to 1 (only Y channel)

  * **2**: Tensor of type *T* representing U plane. **Required.** Dimensions:

    * ``N`` - batch dimension. Shall be the same as the batch dimension for Y plane
    * ``H`` - height dimension shall be half of the image height (for example, ``image_height / 2``)
    * ``W`` - width dimension shall be half of the image width (for example, ``image_width / 2``)
    * ``C`` - channels dimension shall be equal to 1 (U channel)

  * **3**: Tensor of type *T* representing V plane. **Required.** Dimensions:

    * ``N`` - batch dimension. Shall be the same as the batch dimension for Y plane
    * ``H`` - height dimension shall be half of the image height (for example, ``image_height / 2``)
    * ``W`` - width dimension shall be half of the image width (for example, ``image_width / 2``)
    * ``C`` - channels dimension shall be equal to 1 (V channel)

**Outputs:**

* **1**: A tensor of type *T* representing an image converted in RGB format. Dimensions:

  * ``N`` - batch dimension
  * ``H`` - height dimension is the same as the image height
  * ``W`` - width dimension is the same as the image width
  * ``C`` - channels dimension is equal to 3. The first channel is Red, the second one is Green, the last one is Blue

**Types:**

* *T*: ``uint8`` or any supported floating-point type.


**Examples:**

*Example 1*

.. code-block:: xml
   :force:

   <layer ... type="I420toRGB">
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>720</dim>
               <dim>640</dim>
               <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="1">
               <dim>1</dim>
               <dim>480</dim>
               <dim>640</dim>
               <dim>3</dim>
           </port>
       </output>
   </layer>


*Example 2*

.. code-block:: xml
   :force:

   <layer ... type="I420toRGB">
       <input>
           <port id="0">  <!-- Y plane -->
               <dim>1</dim>
               <dim>480</dim>
               <dim>640</dim>
               <dim>1</dim>
           </port>
           <port id="1">  <!-- U plane -->
               <dim>1</dim>
               <dim>240</dim>
               <dim>320</dim>
               <dim>1</dim>
           </port>
           <port id="2">  <!-- V plane -->
             <dim>1</dim>
             <dim>240</dim>
             <dim>320</dim>
             <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="1">
               <dim>1</dim>
               <dim>480</dim>
               <dim>640</dim>
               <dim>3</dim>
           </port>
       </output>
   </layer>



