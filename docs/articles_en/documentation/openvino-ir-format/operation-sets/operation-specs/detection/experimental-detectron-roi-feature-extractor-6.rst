ExperimentalDetectronROIFeatureExtractor
========================================


.. meta::
  :description: Learn about ExperimentalDetectronROIFeatureExtractor-6 -
                an object detection operation, which can be performed on two
                required input tensors.

**Versioned name**: *ExperimentalDetectronROIFeatureExtractor-6*

**Category**: *Object detection*

**Short description**: *ExperimentalDetectronROIFeatureExtractor* is the :doc:`ROIAlign <roi-align-3>` operation applied over a feature pyramid.

**Detailed description**: *ExperimentalDetectronROIFeatureExtractor* maps input ROIs to the levels of the pyramid depending on the sizes of ROIs and parameters of the operation, and then extracts features via ROIAlign from corresponding pyramid levels.

Operation applies the *ROIAlign* algorithm to the pyramid layers:

``output[i, :, :, :] = ROIAlign(inputPyramid[j], rois[i])``

``j = PyramidLevelMapper(rois[i])``

PyramidLevelMapper maps the ROI to the pyramid level using the following formula:

``j = floor(2 + log2(sqrt(w * h) / 224)``

Here 224 is the canonical ImageNet pre-training size, 2 is the pyramid starting level, and ``w``, ``h`` are the ROI width and height.

For more details please see the following source: `Feature Pyramid Networks for Object Detection <https://arxiv.org/pdf/1612.03144.pdf>`__.

**Attributes**:

* *output_size*

  * **Description**: The *output_size* attribute specifies the width and height of the output tensor.
  * **Range of values**: a positive integer number
  * **Type**: ``int``
  * **Default value**: None
  * **Required**: *yes*

* *sampling_ratio*

  * **Description**: The *sampling_ratio* attribute specifies the number of sampling points per the output value. If 0, then use adaptive number computed as ``ceil(roi_width / output_width)``, and likewise for height.
  * **Range of values**: a non-negative integer number
  * **Type**: ``int``
  * **Default value**: None
  * **Required**: *yes*

* *pyramid_scales*

  * **Description**: The *pyramid_scales* enlists ``image_size / layer_size[l]`` ratios for pyramid layers ``l=1,...,L``, where ``L`` is the number of pyramid layers, and ``image_size`` refers to network's input image. Note that pyramid's largest layer may have smaller size than input image, e.g. ``image_size`` is ``800 x 1344`` in the XML example below.
  * **Range of values**: a list of positive integer numbers
  * **Type**: ``int[]``
  * **Default value**: None
  * **Required**: *yes*

* *aligned*

  * **Description**: The *aligned* attribute specifies add offset (``-0.5``) to ROIs sizes or not.
  * **Range of values**:

    * ``true`` - add offset to ROIs sizes
    * ``false`` - do not add offset to ROIs sizes
  * **Type**: boolean
  * **Default value**: false
  * **Required**: *no*

**Inputs**:

* **1**: 2D input tensor of type *T* with shape ``[number_of_ROIs, 4]`` providing the ROIs as 4-tuples: .. math:: [x_{1}, y_{1}, x_{2}, y_{2}]. Coordinates *x* and *y* are refer to the network's input *image_size*. **Required.**
* **2**, ..., **L**: Pyramid of 4D input tensors with feature maps. Shape must be ``[1, number_of_channels, layer_size[l], layer_size[l]]``. The number of channels must be the same for all layers of the pyramid. The layer width and height must equal to the ``layer_size[l] = image_size / pyramid_scales[l]``. **Required.**

**Outputs**:

* **1**: 4D output tensor of type *T* with ROIs features. Shape must be ``[number_of_ROIs, number_of_channels, output_size, output_size]``. Channels number is the same as for all images in the input pyramid.
* **2**: 2D output tensor of type *T* with reordered ROIs according to their mapping to the pyramid levels. Shape must be the same as for 1 input: ``[number_of_ROIs, 4]``.

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="ExperimentalDetectronROIFeatureExtractor" version="opset6">
       <data aligned="false" output_size="7" pyramid_scales="4,8,16,32,64" sampling_ratio="2"/>
       <input>
           <port id="0">
               <dim>1000</dim>
               <dim>4</dim>
           </port>
           <port id="1">
               <dim>1</dim>
               <dim>256</dim>
               <dim>200</dim>
               <dim>336</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>256</dim>
               <dim>100</dim>
               <dim>168</dim>
           </port>
           <port id="3">
               <dim>1</dim>
               <dim>256</dim>
               <dim>50</dim>
               <dim>84</dim>
           </port>
           <port id="4">
               <dim>1</dim>
               <dim>256</dim>
               <dim>25</dim>
               <dim>42</dim>
           </port>
       </input>
       <output>
           <port id="5" precision="FP32">
               <dim>1000</dim>
               <dim>256</dim>
               <dim>7</dim>
               <dim>7</dim>
           </port>
           <port id="6" precision="FP32">
               <dim>1000</dim>
               <dim>4</dim>
           </port>
       </output>
   </layer>


