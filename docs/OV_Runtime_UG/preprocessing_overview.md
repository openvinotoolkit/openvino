# Using Preprocessing API {#openvino_docs_OV_Runtime_UG_Preprocessing}

## Using Preprocessing API

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective


When your input data don't perfectly fit to Neural Network model input tensor - this means that additional operations/steps are needed to transform your data to format expected by model. These operations are known as "preprocessing".

### Example
Consider the following standard example: model expects input with shape `{1, 3, 224, 224}`, `FP32` precision, RGB channels order, and require data normalization (subtract mean and divide by scale factor). But you have BGR 640x480 image (data is `{480, 640, 3})`. This means that we need some operations which will:
- Convert U8 buffer to FP32
- Transform to `planar` format: from `{1, 480, 640, 3}` to `{1, 3, 480, 640}`
- Resize image from 640x480 to 224x224
- Make it `BGR->RGB` conversion as model expects `RGB`
- For each pixel, subtract mean values and divide by scale factor.

![preprocess_not_fit]

Even though all these steps can be relatively easy implemented manually in application's code before actual inference, it is possible to do it with PreProcessing API. Reasons to use this API are:
- PreProcessing API is easy to use
- Preprocessing steps will be integrated into execution graph and will be performed on selected device (CPU/GPU/VPU/etc.) rather than always being executed on CPU. This will improve selected device utilization which is always good.

## Preprocessing API

Intuitively, Preprocessing API consists of the following parts:
1) **Tensor:** Declare user's data format, like shape, layout, precision, color format of actual user's data
2) **Steps:** Describe sequence of preprocessing steps which need to be applied to user's data
3) **Model:** Specify Model's data format. Usually, precision and shape are already known for model, only additional information, like `layout` can be specified

**Note:** All model's graph modification shall be performed after model is read from disk and **before** it is being loaded on actual device. See also [OpenVINO™ Common Inference pipeline](../migration_ov_2_0/docs/common_inference_pipeline.md)

### PrePostProcessor object

`ov::preprocess::PrePostProcessor` class allows specifying preprocessing and postprocessing steps for model read from disk.

@snippet snippets/ov_preprocessing.cpp ov:preprocess:create

### Declare user's data format

To address particular input of model/preprocessor, use `PrePostProcessor::input(...)` method

@snippet snippets/ov_preprocessing.cpp ov:preprocess:tensor

Here we've specified all information about user's input:
 - Precision is U8 (unsigned 8-bit integer)
 - Data represents tensor with {1,480,640,3} shape
 - Layout is "NHWC". It means that 'height=480, width=640, channels=3'
 - Color format is `BGR`

### Declare model's layout

Model's input already has information about precision and shape. Preprocessing API is not intended to modify this. The only thing that may be specified is input's data layout

@snippet snippets/ov_preprocessing.cpp ov:preprocess:model

Now, if model's input has `{1,3,224,224}` shape, preprocessing will be able to identify that model's `height=224`, `width=224`, `channels=3`. Height/width information is necessary for 'resize', and `channels` is needed for mean/scale normalization

### Preprocessing steps

Now we can define sequence of preprocessing steps:

@snippet snippets/ov_preprocessing.cpp ov:preprocess:steps

Here:
 - Convert U8 to FP32 precision
 - Convert current color format (BGR) to RGB
 - Resize to model's height/width. **Note** that if model accepts dynamic size, e.g. {?, 3, ?, ?}, `resize` will not know how to resize the picture, so in this case you should specify target height/width on this step. See [TBD](@ref)
 - Subtract mean from each channel. On this step, color format is RGB already, so `100.5` will be subtracted from each Red component, and `101.5` will be subtracted from `Blue` one.
 - Divide each pixel data to appropriate scale value. In this example each `Red` component will be divided by 50, `Green` by 51, `Blue` by 52 respectively
 - **Note:** last `convert_layout` step is commented out as it is not necessary to specify last layout conversion. PrePostProcessor will do such conversion automatically

### Integrate steps into model

We've finished with preprocessing steps declaration, now it is time to build it. For debugging purposes it is possible to print `PrePostProcessor` configuration on screen:

@snippet snippets/ov_preprocessing.cpp ov:preprocess:steps

After this, `model` will accept U8 input with {1, 480, 640, 3} shape, with BGR channels order. All conversion steps will be integrated into execution graph.

## Using Preprocessing API (Python)

@sphinxdirective
.. raw:: html

    <div id="switcher-python" class="switcher-anchor">Python</div>
@endsphinxdirective

[preprocess_not_fit]: img/preprocess_not_fit.png
