ExperimentalDetectronPriorGridGenerator
=======================================


.. meta::
  :description: Learn about ExperimentalDetectronPriorGridGenerator-6 -
                an object detection operation, which can be performed on three
                required input tensors.

**Versioned name**: *ExperimentalDetectronPriorGridGenerator-6*

**Category**: *Object detection*

**Short description**: The *ExperimentalDetectronPriorGridGenerator* operation generates prior grids of specified sizes.

**Detailed description**: The operation takes coordinates of centres of boxes and adds strides with offset `0.5` to them to calculate coordinates of prior grids.

Numbers of generated cells is ``featmap_height`` and ``featmap_width`` if *h* and *w* are zeroes; otherwise, *h* and *w*, respectively. Steps of generated grid are ``image_height`` / ``layer_height`` and ``image_width`` / ``layer_width`` if *stride_h* and *stride_w* are zeroes; otherwise, *stride_h* and *stride_w*, respectively.

``featmap_height``, ``featmap_width``, ``image_height`` and ``image_width`` are spatial dimensions values from second and third inputs, respectively.

**Attributes**:

* *flatten*

  * **Description**: The *flatten* attribute specifies whether the output tensor should be 2D or 4D.
  * **Range of values**:

    * ``true`` - the output tensor should be a 2D tensor
    * ``false`` - the output tensor should be a 4D tensor
  * **Type**: ``boolean``
  * **Default value**: true
  * **Required**: *no*

* *h*

  * **Description**: The *h* attribute specifies number of cells of the generated grid with respect to height.
  * **Range of values**: non-negative integer number less or equal than ``featmap_height``
  * **Type**: ``int``
  * **Default value**: 0
  * **Required**: *no*

* *w*

  * **Description**: The *w* attribute specifies number of cells of the generated grid with respect to width.
  * **Range of values**: non-negative integer number less or equal than ``featmap_width``
  * **Type**: ``int``
  * **Default value**: 0
  * **Required**: *no*

* *stride_x*

  * **Description**: The *stride_x* attribute specifies the step of generated grid with respect to x coordinate.
  * **Range of values**: non-negative float number
  * **Type**: ``float``
  * **Default value**: 0.0
  * **Required**: *no*

* *stride_y*

  * **Description**: The *stride_y* attribute specifies the step of generated grid with respect to y coordinate.
  * **Range of values**: non-negative float number
  * **Type**: ``float``
  * **Default value**: 0.0
  * **Required**: *no*

**Inputs**

* **1**: A 2D tensor of type *T* with shape ``[number_of_priors, 4]`` contains priors. **Required.**
* **2**: A 4D tensor of type *T* with input feature map ``[1, number_of_channels, featmap_height, featmap_width]``. This operation uses only sizes of this input tensor, not its data. **Required.**
* **3**: A 4D tensor of type *T* with input image ``[1, number_of_channels, image_height, image_width]``. The number of channels of both feature map and input image tensors must match. This operation uses only sizes of this input tensor, not its data. **Required.**

**Outputs**

* **1**: A tensor of type *T* with priors grid with shape ``[featmap_height * featmap_width * number_of_priors, 4]`` if flatten is ``true`` or ``[featmap_height, featmap_width, number_of_priors, 4]``, otherwise. If 0 < *h* < ``featmap_height`` and/or 0 < *w* < ``featmap_width`` the output data size is less than ``featmap_height`` * ``featmap_width`` * ``number_of_priors`` * 4 and the output tensor is filled with undefined values for rest output tensor elements.

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="ExperimentalDetectronPriorGridGenerator" version="opset6">
       <data flatten="true" h="0" stride_x="32.0" stride_y="32.0" w="0"/>
       <input>
           <port id="0">
               <dim>3</dim>
               <dim>4</dim>
           </port>
           <port id="1">
               <dim>1</dim>
               <dim>256</dim>
               <dim>25</dim>
               <dim>42</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>3</dim>
               <dim>800</dim>
               <dim>1344</dim>
           </port>
       </input>
       <output>
           <port id="3" precision="FP32">
               <dim>3150</dim>
               <dim>4</dim>
           </port>
       </output>
   </layer>


