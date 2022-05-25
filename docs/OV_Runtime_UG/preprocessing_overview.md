# Optimize Preprocessing {#openvino_docs_OV_UG_Preprocessing_Overview}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_Preprocessing_Details
   openvino_docs_OV_UG_Layout_Overview
   openvino_docs_OV_UG_Preprocess_Usecase_save

@endsphinxdirective

## Introduction

When the input data does not fit to Neural Network model input tensor perfectly, this means that additional operations/steps are needed to transform the data to format expected by a model. These operations are known as "preprocessing".

### Example
Consider the following standard example: deep learning model expects input with the `{1, 3, 224, 224}` shape, `FP32` precision, `RGB` color channels order, and it requires data normalization (subtract mean and divide by scale factor). However, there is just a `640x480` `BGR` image (data is `{480, 640, 3}`). This means that operations below must be performed:
 - Convert `U8` buffer to `FP32`.
 - Transform to `planar` format: from `{1, 480, 640, 3}` to `{1, 3, 480, 640}`.
 - Resize image from 640x480 to 224x224.
 - Make `BGR->RGB` conversion as model expects `RGB`.
 - For each pixel, subtract mean values and divide by scale factor.


![](img/preprocess_not_fit.png)


Even though all these steps can be relatively easy to implement manually in the application code before actual inference, it is possible to do it with Preprocessing API. Advantages of using this API are:
 - Preprocessing API is easy to use.
 - Preprocessing steps will be integrated into execution graph and will be performed on selected device (CPU/GPU/VPU/etc.) rather than always being executed on CPU. This will improve selected device utilization which is always good.

## Preprocessing API

Intuitively, preprocessing API consists of the following parts:
 1. 	**Tensor** -- It declares user data format, like shape, [layout](./layout_overview.md), precision, color format from actual user's data.
 2. 	**Steps** - It describes sequence of preprocessing steps which need to be applied to user data.
 3. 	**Model** - It specifies model data format. Usually, precision and shape are already known for model, only additional information, like [layout](./layout_overview.md) can be specified.

> **Note:** Graph modification of a model shall be performed after a model is read from a disk and **before** it is loaded on actual device.

### PrePostProcessor Object

The `ov::preprocess::PrePostProcessor` class allows specifying preprocessing and postprocessing steps for a model read from disk.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:create

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:create

@endsphinxtab

@endsphinxtabset

### Declare User's Data Format

To address particular input of a model/preprocessor, use `ov::preprocess::PrePostProcessor::input(input_name)` method.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:tensor

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:tensor

@endsphinxtab

@endsphinxtabset

Below is all the specified input information:
 - Precision is `U8` (unsigned 8-bit integer).
 - Data represents tensor with the `{1,480,640,3}` shape.
 - [Layout](./layout_overview.md) is "NHWC". It means: `height=480`, `width=640`, `channels=3`'.
 - Color format is `BGR`.

@anchor declare_model_s_layout
### Declaring Model Layout

Model input already has information about precision and shape. Preprocessing API is not intended to modify this. The only thing that may be specified is input data [layout](./layout_overview.md)

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:model

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:model

@endsphinxtab

@endsphinxtabset


Now, if model input has the `{1,3,224,224}` shape, preprocessing will be able to identify the `height=224`, `width=224`, and `channels=3` of that model. The `height`/`width` information is necessary for `resize`, and `channels` is needed for mean/scale normalization.

### Preprocessing Steps

Now, define sequence of preprocessing steps:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:steps

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:steps

@endsphinxtab

@endsphinxtabset

Perform as follows:

   1. Convert `U8` to `FP32` precision.
   2. Convert current color format (`BGR`) to `RGB`.
   3. Resize to `height`/`width` of a model. Be aware that if a model accepts dynamic size, e.g. `{?, 3, ?, ?}`, `resize` will not know how to resize the picture. Therefore, in this case, target `height`/`width` should be specified. See also `ov::preprocess::PreProcessSteps::resize()`.
   4. Subtract mean from each channel. In this step, color format is RGB already, so `100.5` will be subtracted from each `Red` component, and `101.5` will be subtracted from `Blue` one.
   5. Divide each pixel data to appropriate scale value. In this example, each `Red` component will be divided by 50, `Green` by 51, `Blue` by 52 respectively.
   6. Note that the last `convert_layout` step is commented out as it is not necessary to specify the last layout conversion. PrePostProcessor will do such conversion automatically.

### Integrating Steps into a Model

Build it when the preprocessing has been finished. It is possible to print `PrePostProcessor` configuration on screen for debugging purposes:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:build

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:build

@endsphinxtab

@endsphinxtabset


After this, a `model` will accept `U8` input with `{1, 480, 640, 3}` shape, with `BGR` channels order. All conversion steps will be integrated into execution graph. Now, load the model on the device and pass the image to the model as is, without any data manipulation in the application.


## See Also

* [Preprocessing Details](./preprocessing_details.md)
* [Layout API overview](./layout_overview.md)
* <code>ov::preprocess::PrePostProcessor</code> C++ class documentation
