# Dynamic Shapes {#openvino_docs_OV_UG_DynamicShapes}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_NoDynamicShapes

@endsphinxdirective

As it was demonstrated in the [Changing Input Shapes](ShapeInference.md) article, there are models that support changing of input shapes before model compilation in `Core::compile_model`.
Reshaping models provides an ability to customize the model input shape for exactly the size that is required in the end application.
This article explains how the ability of a model to reshape can further be leveraged in more dynamic scenarios.

## When to Apply Dynamic Shapes

Conventional "static" model reshaping works well when it can be done once per many model inference calls with the same shape.
However, this approach does not perform efficiently if the input tensor shape is changed on every inference call: calling `reshape()` and `compile_model()` each time when a new size comes is extremely time-consuming.
A popular example would be an inference of natural language processing models (like BERT) with arbitrarily-sized input sequences that come from a user.
In this case, the sequence length cannot be predicted and may change every time you need to call inference.
Below, such dimensions that can be frequently changed are called *dynamic dimensions*.
Dynamic shapes should be considered when the real shape of input is not known at `compile_model` time.

Here are several examples of dimensions that can be naturally dynamic:
 - Sequence length dimension for various sequence processing models, like BERT
 - Spatial dimensions in segmentation and style transfer models
 - Batch dimension
 - Arbitrary number of detections in object detection models output

There are various tricks to address input dynamic dimensions through combining multiple pre-reshaped models and input data padding.
The tricks are sensitive to model internals. They do not always give optimal performance and are cumbersome.
You will find a short overview of the methods [here](ov_without_dynamic_shapes.md).
Apply those methods only if the native dynamic shape API described in the following sections does not work for you or does not perform as desired.

The decision about using dynamic shapes should be based on proper benchmarking of a real application with a real data.
For unlike statically shaped models, inference of dynamically shaped ones takes different inference time depending on input data shape or input tensor content.
Also, using the dynamic shapes can bring more overheads in memory and running time per each inference call, depending on a hardware plugin and a used model.

## Dynamic Shapes without Tricks

This section describes how to handle dynamically shaped models natively with OpenVINO Runtime API version 2022.1 and higher.
There are three main parts in the flow that differ from static shapes:
 - configure the model
 - prepare data for inference
 - read resulting data after inference

### Configure the Model

To avoid the tricks mentioned in the previous section, there is a way to directly specify one or multiple dimensions in the model inputs to be dynamic.
This is achieved with the same reshape method that is used for alternating static shape of inputs.
Dynamic dimensions are specified as `-1` or `ov::Dimension()` instead of a positive number used for static dimensions:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:reshape_undefined

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_dynamic_shapes.py reshape_undefined
@endsphinxtab

@endsphinxtabset

To simplify the code, the examples assume that the model has a single input and single output.
However, there are no limitations on the number of inputs and outputs to apply dynamic shapes.

### Undefined Dimensions "Out Of the Box"

Dynamic dimensions may appear in the input model without calling reshape.
Many DL frameworks support undefined dimensions.
If such a model is converted with Model Optimizer or read directly by Core::read_model, undefined dimensions are preserved.
Such dimensions automatically treated as dynamic ones.
So you don't need to call reshape if undefined dimensions are already configured in the original model or in the IR file.

If the input model has undefined dimensions that you are not going to change during the inference, it is recommended to set them to static values, using the same `reshape` method of the model.
From the API perspective any combination of dynamic and static dimensions can be configured.

Model Optimizer provides identical capability to reshape the model during the conversion, including specifying dynamic dimensions.
Use this capability to save time on calling `reshape` method in the end application.
To get information about setting input shapes using Model Optimizer, refer to [Setting Input Shapes](../MO_DG/prepare_model/convert_model/Converting_Model.md)

### Dimension Bounds

Besides marking a dimension just dynamic, you can also specify lower and/or upper bounds that define a range of allowed values for the dimension.
Bounds are coded as arguments for `ov::Dimension`:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:reshape_bounds

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_dynamic_shapes.py reshape_bounds
@endsphinxtab
@endsphinxtabset

Information about bounds gives opportunity for the inference plugin to apply additional optimizations.
Using dynamic shapes assumes the plugins apply more loose optimization technique during model compilation
It may require more time/memory for model compilation and inference.
So providing any additional information like bounds can be beneficial.
For the same reason it is not recommended to leave dimensions as undefined without the real need.

When specifying bounds, the lower bound is not so important as upper bound, because knowing of upper bound allows inference devices to more precisely allocate memory for intermediate tensors for inference and use lesser number of tuned kernels for different sizes.
Precisely speaking benefits of specifying lower or upper bound is device dependent.
Depending on the plugin specifying upper bounds can be required. For information about dynamic shapes support on different devices, see the [Features Support Matrix](@ref features_support_matrix).

If users known lower and upper bounds for dimension it is recommended to specify them even when plugin can execute model without the bounds.

### Setting Input Tensors

Preparing model with the reshape method was the first step.
The second step is passing a tensor with an appropriate shape to infer request.
This is similar to the [regular steps](integrate_with_your_application.md), but now you can pass tensors with different shapes for the same executable model and even for the same inference request:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:set_input_tensor

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_dynamic_shapes.py set_input_tensor
@endsphinxtab
@endsphinxtabset

In the example above, `set_input_tensor` is used to specify input tensors.
The real dimensions of the tensor are always static, because it is a concrete tensor and it does not have any dimension variations in contrast to model inputs.

Similar to static shapes, `get_input_tensor` can be used instead of `set_input_tensor`.
In contrast to static input shapes, when using `get_input_tensor` for dynamic inputs, `set_shape` method for the returned tensor should be called to define the shape and allocate memory.
Without doing that, the tensor returned by `get_input_tensor` is an empty tensor. Its shape is not initialized and memory is not allocated, because infer request does not have information about real shape you are going to feed.
Setting shape for input tensor is required when the corresponding input has at least one dynamic dimension regardless of bounds information.
The following example makes the same sequence of two infer requests as the previous example but using `get_input_tensor` instead of `set_input_tensor`:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:get_input_tensor

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_dynamic_shapes.py get_input_tensor
@endsphinxtab

@endsphinxtabset

### Dynamic Shapes in Outputs

Examples above handle correctly the case when dynamic dimensions in output may be implied by propagating of dynamic dimension from the inputs.
For example, batch dimension in input shape is usually propagated through the whole model and appears in the output shape.
The same is true for other dimensions, like sequence length for NLP models or spatial dimensions for segmentation models, that are propagated through the entire network.

Whether output has dynamic dimensions or not can be examined by querying output partial shape after model read or reshape.
The same is applicable for the inputs. For example:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:print_dynamic

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_dynamic_shapes.py print_dynamic
@endsphinxtab

@endsphinxtabset

Appearing `?` or ranges like `1..10` means there are dynamic dimensions in corresponding inputs or outputs.

Or more programmatically:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:detect_dynamic

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_dynamic_shapes.py detect_dynamic
@endsphinxtab

@endsphinxtabset


If at least one dynamic dimension exists in output of the model, shape of the corresponding output tensor will be set as the result of inference call.
Before the first inference, memory for such a tensor is not allocated and has shape `[0]`.
If a user calls `set_output_tensor` with pre-allocated tensor, the inference will call `set_shape` internally, and the initial shape is replaced by the calculated shape.
Setting shape for output tensors, in this case, is useful only if you want to pre-allocate enough memory for an output tensor. The `set_shape` method of the `Tensor` will re-allocate memory only if a new shape requires more storage.
