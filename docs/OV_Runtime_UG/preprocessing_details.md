# Preprocessing API - Details {#openvino_docs_OV_UG_Preprocessing_Details}

The purpose of this article is to present details on preprocessing API, such as its capabilities and post-processing.

## Preprocessing Capabilities

### Addressing Particular Input/Output

If the model has only one input, then simple `ov::preprocess::PrePostProcessor::input()` will get a reference to preprocessing builder for this input (a tensor, the steps, a model):

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:input_1

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:input_1

@endsphinxtab

@endsphinxtabset


In general, when a model has multiple inputs/outputs, each one can be addressed by a tensor name.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:input_name

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:input_name

@endsphinxtab

@endsphinxtabset


Or by it's index.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:input_index

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:input_index

@endsphinxtab

@endsphinxtabset

C++ references:
  * `ov::preprocess::InputTensorInfo`
  * `ov::preprocess::OutputTensorInfo`
  * `ov::preprocess::PrePostProcessor`


### Supported Preprocessing Operations

C++ references:
* `ov::preprocess::PreProcessSteps`

#### Mean/Scale Normalization

Typical data normalization includes 2 operations for each data item: subtract mean value and divide to standard deviation. This can be done with the following code:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:mean_scale

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:mean_scale

@endsphinxtab

@endsphinxtabset

In Computer Vision area normalization is usually done separately for R, G, B values. To do this, [layout with 'C' dimension](./layout_overview.md) shall be defined. Example:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:mean_scale_array

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:mean_scale_array

@endsphinxtab

@endsphinxtabset

C++ references:
* `ov::preprocess::PreProcessSteps::mean()`
* `ov::preprocess::PreProcessSteps::scale()`


#### Converting Precision

In Computer Vision, image is represented by an array of unsigned 8-but integer values (for each color), but model accepts floating point tensors.

To integrate precision conversion into an execution graph as a preprocessing step, simply do the following:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:convert_element_type

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:convert_element_type

@endsphinxtab

@endsphinxtabset

C++ references:
  * `ov::preprocess::InputTensorInfo::set_element_type()`
  * `ov::preprocess::PreProcessSteps::convert_element_type()`


#### Converting layout (transposing)

Transposing of matrices/tensors is a typical operation in Deep Learning - you may have a BMP image 640x480, which is an array of `{480, 640, 3}` elements, but Deep Learning model can require input with shape `{1, 3, 480, 640}`.

Conversion can be done implicitly, using the [layout](./layout_overview.md) of a user's tensor and the layout of an original model.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:convert_layout

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:convert_layout

@endsphinxtab

@endsphinxtabset


For a manual transpose of axes without the use of a [layout](./layout_overview.md) in the code, simply do the following:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:convert_layout_2

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:convert_layout_2

@endsphinxtab

@endsphinxtabset

It performs the same transpose. However, the approach where source and destination layout are used can be easier to read and understand.

C++ references:
  * `ov::preprocess::PreProcessSteps::convert_layout()`
  * `ov::preprocess::InputTensorInfo::set_layout()`
  * `ov::preprocess::InputModelInfo::set_layout()`
  * `ov::Layout`

#### Resizing Image

Resizing of an image is a typical preprocessing step for computer vision tasks. With preprocessing API, this step can also be integrated into execution graph and performed on a target device.

To resize the input image, it is needed to define `H` and `W` dimensions of the [layout](./layout_overview.md)

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:resize_1

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:resize_1

@endsphinxtab

@endsphinxtabset

When original model has known spatial dimensions (`width`+`height`), target `width`/`height` can be omitted.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:resize_2

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:resize_2

@endsphinxtab

@endsphinxtabset

C++ references:
* `ov::preprocess::PreProcessSteps::resize()`
* `ov::preprocess::ResizeAlgorithm`


#### Color Conversion

Typical use case is to reverse color channels from `RGB` to `BGR` and vice versa. To do this, specify source color format in `tensor` section and perform `convert_color` preprocessing operation. In the example below, a `BGR` image needs to be converted to `RGB` as required for the model input.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:convert_color_1

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:convert_color_1

@endsphinxtab

@endsphinxtabset

#### Color Conversion - NV12/I420
Preprocessing also supports YUV-family source color formats, i.e. NV12 and I420.
In advanced cases, such YUV images can be split into separate planes, e.g. for NV12 images Y-component may come from one source and UV-component from another one. Concatenating such components in user's application manually is not a perfect solution from performance and device utilization perspectives. However, there is a way to use Preprocessing API. For such cases there are `NV12_TWO_PLANES` and `I420_THREE_PLANES` source color formats, which will split the original `input` into 2 or 3 inputs.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:convert_color_2

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:convert_color_2

@endsphinxtab

@endsphinxtabset

In this example, the original `input` is split to `input/y` and `input/uv` inputs. You can fill `input/y` from one source, and `input/uv` from another source. Color conversion to `RGB` will be performed, using these sources. It is more efficient as there will be no additional copies of NV12 buffers.

C++ references:
* `ov::preprocess::ColorFormat`
* `ov::preprocess::PreProcessSteps::convert_color`


### Custom Operations

Preprocessing API also allows adding custom preprocessing steps into execution graph. Custom step is a function, which accepts current `input` node and returns new node after adding preprocessing step.

> **Note:** Custom preprocessing function should only insert node(s) after the input, it will be done during model compilation. This function will NOT be called during execution phase. This may seem serious and require some knowledge of [OpenVINO™ operations](../ops/opset.md).

If there is a need to insert some additional operations to the execution graph right after the input, like some specific crops and/or resizes - Preprocessing API can be a good choice to implement this.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:custom

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:custom

@endsphinxtab

@endsphinxtabset

C++ references:
* `ov::preprocess::PreProcessSteps::custom()`
* [Available Operations Sets](../ops/opset.md)

## Postprocessing

Post-processing steps can be added to model outputs. As for pre-processing, these steps will be also integrated into a graph and executed on a selected device.

Pre-processing uses the following flow: **User tensor** -> **Steps** -> **Model input**.

Post-processing uses the reverse: **Model output** -> **Steps** -> **User tensor**.

Compared to pre-processing, there are not so many operations needed for post-processing stage. Currently, only the following post-processing operations are supported:
 - Convert a [layout](./layout_overview.md).
 - Convert an element type.
 - Customize operations.

Usage of these operations is similar to pre-processing. Below is an example:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:postprocess

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:postprocess

@endsphinxtab

@endsphinxtabset

C++ references:
* `ov::preprocess::PostProcessSteps`
* `ov::preprocess::OutputModelInfo`
* `ov::preprocess::OutputTensorInfo`
