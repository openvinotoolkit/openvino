# Preprocessing API - details {#openvino_docs_OV_Runtime_UG_Preprocessing_Details}

## Preprocessing capabilities

### Addressing particular input/output

If your model has only one input, then simple <code>ov::preprocess::PrePostProcessor::input()</code> will get a reference to preprocessing builder for this input (tensor, steps, model):

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:input_1]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:input_1]

@endsphinxdirective

In general, when model has multiple inputs/outputs, each one can be addressed by name

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:input_name]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:input_name]

@endsphinxdirective


Or by it's index

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:input_index]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:input_index]

@endsphinxdirective

C++ references:
  * <code>ov::preprocess::InputTensorInfo</code>
  * <code>ov::preprocess::OutputTensorInfo</code>
  * <code>ov::preprocess::PrePostProcessor</code>


### Supported preprocessing operations

C++ references:
* <code>ov::preprocess::PreProcessSteps</code>

#### Mean/Scale normalization

Typical data normalization includes 2 operations for each data item: subtract mean value and divide to standard deviation. This can be done with the following code:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:mean_scale]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:mean_scale]

@endsphinxdirective


In Computer Vision area normalization is usually done separately for R, G, B values. To do this, [layout with 'C' dimension](./layout_overview.md) shall be defined. Example:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:mean_scale_array]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:mean_scale_array]

@endsphinxdirective

C++ references:
* <code>ov::preprocess::PreProcessSteps::mean()</code>
* <code>ov::preprocess::PreProcessSteps::scale()</code>


#### Convert precision

In Computer Vision, image is represented by array of unsigned 8-but integer values (for each color), but model accepts floating point tensors

To integrate precision conversion into execution graph as a preprocessing step, just do:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:convert_element_type]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:convert_element_type]

@endsphinxdirective

C++ references:
  * <code>ov::preprocess::InputTensorInfo::set_element_type()</code>
  * <code>ov::preprocess::PreProcessSteps::convert_element_type()</code>


#### Convert layout (transpose)

Transposing of matrices/tensors is a typical operation in Deep Learning - you may have a BMP image 640x480 which is an array of `{480, 640, 3}` elements, but Deep Learning model can require input with shape `{1, 3, 480, 640}`

Using [layout](./layout_overview.md) of user's tensor and layout of original model conversion can be done implicitly

@sphinxdirective
.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:convert_layout]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:convert_layout]

@endsphinxdirective


Or if you prefer manual transpose of axes without usage of [layout](./layout_overview.md) in your code, just do:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:convert_layout_2]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:convert_layout_2]

@endsphinxdirective

It performs the same transpose, but we believe that approach using source and destination layout can be easier to read and understand

C++ references:
  * <code>ov::preprocess::PreProcessSteps::convert_layout()</code>
  * <code>ov::preprocess::InputTensorInfo::set_layout()</code>
  * <code>ov::preprocess::InputModelInfo::set_layout()</code>
  * <code>ov::Layout</code>

#### Resize image

Resizing of image is a typical preprocessing step for computer vision tasks. With preprocessing API this step can also be integrated into execution graph and performed on target device.

To resize the input image, it is needed to define 'H' and 'W' dimensions of [layout](./layout_overview.md)

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:resize_1]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:resize_1]

@endsphinxdirective

Or in case if original model has known spatial dimensions (widht+height), target width/height can be omitted

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:resize_2]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:resize_2]

@endsphinxdirective

C++ references:
* <code>ov::preprocess::PreProcessSteps::resize()</code>
* <code>ov::preprocess::ResizeAlgorithm</code>


#### Color conversion

Typical use case is to reverse color channels from RGB to BGR and wise versa. To do this, specify source color format in `tensor` section and perform `convert_color` preprocessing operation. In example below, user has `BGR` image and needs to convert it to `RGB` as required for model's input

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:convert_color_1]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:convert_color_1]

@endsphinxdirective

#### Color conversion - NV12/I420
Preprocessing also support YUV-family source color formats, i.e. NV12 and I420.
In advanced cases such YUV images can be splitted into separate planes, e.g. for NV12 images Y-component may come from one source and UV-component comes from another source. Concatenating such components in user's application manually is not a perfect solution from performance and device utilization perspectives, so there is a way to use Preprocessing API. For such cases there is `NV12_TWO_PLANES` and `I420_THREE_PLANES` source color formats, which will split original `input` to 2 or 3 inputs

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:convert_color_2]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:convert_color_2]

@endsphinxdirective

In this example, original `input` is being split to `input/y` and `input/uv` inputs. You can fill `input/y` from one source, and `input/uv` from another source. Color conversion to `RGB` will be performed using these sources, it is more optimal as there will be no additional copies of NV12 buffers.

C++ references:
* <code>ov::preprocess::ColorFormat</code>
* <code>ov::preprocess::PreProcessSteps::convert_color</code>


### Custom operations

Preprocessing API also allows adding custom preprocessing steps into execution graph. Custom step is a function which accepts current 'input' node and returns new node after adding preprocessing step

>**Note:** Custom preprocessing function shall only insert node(s) after input, it will be done during model compilation. This function will NOT be called during execution phase. This may look not trivial and require some knowledge of [OpenVINO™ operations](../ops/opset.md)

If there is a need to insert some additional operations to execution graph right after input, like some specific crops and/or resizes - Preprocessing API can be a good choice to implement this

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:custom]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:custom]

@endsphinxdirective

C++ references:
* <code>ov::preprocess::PreProcessSteps::custom()</code>
* [Available Operations Sets](../ops/opset.md)

## Postprocessing

Postprocessing steps can be added to model outputs. As for preprocessing, these steps will be also integrated into graph and executed on selected device.

Preprocessing uses flow **User tensor** -> **Steps** -> **Model input**

Postprocessing is wise versa:  **Model output** -> **Steps** -> **User tensor**

Comparing to preprocessing, there is not so much operations needed to do in post-processing stage, so right now only following postprocessing operations are supported:
 - Convert [layout](./layout_overview.md)
 - Convert element type
 - Custom operations

Usage of these operations is similar to Preprocessing. Some example is shown below:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:postprocess]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:postprocess]

@endsphinxdirective

C++ references:
* <code>ov::preprocess::PostProcessSteps</code>
* <code>ov::preprocess::OutputModelInfo</code>
* <code>ov::preprocess::OutputTensorInfo</code>
