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

When your input data do not perfectly fit to Neural Network model input tensor, additional operations/steps are needed to transform your data to the format expected by a model. These operations are known as "preprocessing".

### Example
Consider the following standard example: deep learning model expects input with shape `{1, 3, 224, 224}`, `FP32` precision, `RGB` color channels order, and requires data normalization (subtract mean and divide by scale factor). However, you have just a `640x480` `BGR` image (data is `{480, 640, 3}`). This means that you need some operations which will:
 - Convert U8 buffer to FP32
 - Transform to `planar` format: from `{1, 480, 640, 3}` to `{1, 3, 480, 640}`
 - Resize image from 640x480 to 224x224
 - Make `BGR->RGB` conversion as model expects `RGB`
 - For each pixel, subtract mean values and divide by scale factor


![](img/preprocess_not_fit.png)


All these steps can be quite easily implemented manually in application's code before actual inference. Nevertheless, it is possible to do it with Preprocessing API. 
The reasons to use this API are:
 - Preprocessing API is easy to use.
 - Preprocessing steps are integrated into execution graph and are performed on selected device (CPU/GPU/VPU/etc.) rather than always being executed on CPU. This improves selected device utilization, which is always good.

## Preprocessing API

Intuitively, Preprocessing API consists of the following parts:
 1. 	**Tensor:** Declare user's data format, like shape, the [layout](./layout_overview.md), precision, color format of actual user's data.
 2. 	**Steps:** Describe sequence of preprocessing steps which need to be applied to user's data.
 3. 	**Model:** Specify model data format. Usually, precision and shape for the model are already known. Only additional information, like the [layout](./layout_overview.md) can be specified.

> **Note:** All graph modification of the model shall be performed after the model is read from a disk and **before** it is loaded on actual device.

### PrePostProcessor Object

`ov::preprocess::PrePostProcessor` class allows specifying preprocessing and postprocessing steps for model read from a disk.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:create

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:create

@endsphinxtab

@endsphinxtabset

### Declare User's Data Format

To address particular input of model/preprocessor, use `ov::preprocess::PrePostProcessor::input(input_name)` method.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:tensor

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:tensor

@endsphinxtab

@endsphinxtabset

All user input information is specified below:
 - Precision is U8 (unsigned 8-bit integer)
 - Data represents tensor with {1,480,640,3} shape
 - [Layout](./layout_overview.md) is "NHWC". It means that 'height=480, width=640, channels=3'
 - Color format is `BGR`

@anchor declare_model_s_layout
### Declare the Model Layout

The model input already has information about precision and shape. Preprocessing API is not intended to modify this. The only thing that may be specified is input data [layout](./layout_overview.md)

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:model

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:model

@endsphinxtab

@endsphinxtabset


Now, if the model input has `{1,3,224,224}` shape, preprocessing will be able to identify `height=224`, `width=224`, and `channels=3` of the model. Height/width information is necessary for `resize`, and `channels` is needed for mean/scale normalization.

### Preprocessing Steps

Now you can define sequence of preprocessing steps:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:steps

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:steps

@endsphinxtab

@endsphinxtabset

Here:
 - Convert U8 to FP32 precision
 - Convert current color format (BGR) to RGB
 - Resize to height/width of the model. **Note** that if the model accepts dynamic size, e.g. {?, 3, ?, ?}, `resize` will not know how to resize the picture, so in this case you should specify target height/width in this step. See also <code>ov::preprocess::PreProcessSteps::resize()</code>.
 - Subtract mean from each channel. In this step, color format is RGB already, so `100.5` will be subtracted from each Red component, and `101.5` will be subtracted from `Blue` one.
 - Divide each pixel data to an appropriate scale value. In this example each `Red` component will be divided by 50, `Green` by 51, `Blue` by 52 respectively
 - **Note:** The last `convert_layout` step is commented out as it is not necessary to specify last layout conversion. The `PrePostProcessor` will do such conversion automatically.

### Integrate Steps into the Model

You have finished with preprocessing steps declaration. Now, it is time to build it. For debugging purposes it is possible to print `PrePostProcessor` configuration on screen:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:build

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:build

@endsphinxtab

@endsphinxtabset


After this, the `model` will accept U8 input with `{1, 480, 640, 3}` shape, with `BGR` channels order. All conversion steps will be integrated into execution graph. 
Now, you can load the model on device and pass your image to the model as is, without manipulating any data in the application.


## See Also

* [Preprocessing Details](./preprocessing_details.md)
* [Layout API overview](./layout_overview.md)
* <code>ov::preprocess::PrePostProcessor</code> C++ class documentation
