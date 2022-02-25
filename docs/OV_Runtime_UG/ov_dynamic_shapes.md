#Dynamic Shapes

As it was demonstrated in Reshaping Models<LINK> chapter there are models that support changing of input shapes before model compilation in Core::compile_model.
Reshaping models provides ability to customize model input shape for exactly that size that is required in the end application.
This chapter explains how the ability of model to reshape can further be exploited in more dynamic scenarios.

##Why Dynamic Shapes?

*This section describes cases where you need dynamic shapes and have to use workarounds in case if dynamic shapes are not supported in the first place.
If you already have enough motivation on that, skip this theoretical part and go directly to API guide <LINK>.*

Model reshaping works well when it can be done once per many model inference calls with the same shape.
However, this approach doesn't perform efficiently if the input tensor shape is changed every inference call.
A popular example would be inference of natural language processing models (like BERT) with arbitrary sized input sequences that come from the user.
In this case, the sequence length cannot be predicted.
Calling reshape and compile_model each time when a new size comes is extremely time-consuming.
Such dimensions that can be frequently changed are called *dynamic dimensions*.

For the BERT model you can use a special input to the model to mask unused elements out.
So, the model can be reshaped for some predefined big sequence length once and compiled once, and then the input tensors are used only partially with mask specifying valid tokens.
This approach is called *padding*.

However, padding is not applicable to every model and every use case.
If the model is not designed to handle dummy element gracefully in padding area,
then the results of inference may be totally scrambled,
or accuracy is significantly affected.
Model can even crash during inference.
That means developer should be aware of model internals to apply padding.

Besides the bad developer experience,
the main disadvantage of padding is bad performance due to spending time for processing dummy elements in the padding area,
even if the model is properly designed to be used with padding.
It turns out that usually such models are designed in a way where calculations in padded area still happen not affecting the end result.

Another approach to handle arbitrary sized inputs is to pre-compile several models reshaped for different input shapes.
Works well if the number of different shapes is small enough to afford increased time for multiple reshapes and compilations
as well as increased amount of consumed memory.
As this method cannot be scaled well it is used in combination with the padding:
model with the most suitable input shape among pre-reshaped models is chosen.
It gives smaller pad area in comparison to a single model.

Let's mention another practical but still complicated approach when input tensor can be divided into multiple chunks along dynamic dimension.
For example, if we have a batch of independent inputs as a single tensor.
If arbitrary division along batch dimension is really possible -- and for batch dimension it should be possible by the dimension purpose --
we can run multiple inferences using approach with several pre-compiled models choosing sized to have minimal number of inferences
having a particular batch size in input tensor.

Say if there are models pre-compiled for batch sizes 1, 2, 4 and 8,
input tensor with batch 5 can be processed with two infer calls with batch size 1 and 4.
(Here we assume the batch processing is required for performance reasons otherwise we can just loop over images in a batch,
and process image by image with a single compiled model.)

##Dynamic Shapes without Tricks

In this section we describe how to handle dynamically shaped models more naturally with OpenVINO 2.0 API.
To avoid the tricks described in the previous section there is a way to directly say that one or multiple dimensions in model input is going to be dynamic.
This is achieved with the same reshape method that is used for alternating static shape of inputs.
Dynamic dimensions are marked with `-1` or `ov::Dimension()`.
Both means the same: undefined dimension.

@snippet snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:reshape_undefined

In this and following examples we assume that the model has a single input and single output that let us simplify the code.
There are no restrictions on the number of inputs and output to apply dynamic shapes in the API though.

Dynamic dimensions can exist in the input model without calling reshape.
Many DL frameworks support undefined dimensions.
If such a model is converted with Model Optimizer or read directly by Core::read_model, undefined dimensions are preserved.
So we don't need to call reshape if undefined dimensions are already configured in a desired form in the original model or IR file.

Besides marking a dimension just dynamic, we also can specify bounds for the dimension: lower bound and/or upper bound.
This is a range of allowed values for the dimension.
Bound are coded as arguments for `ov::Dimension`:

@snippet snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:reshape_bounds

Lower bound is not so important as upper bound, because knowing of upper bound allows plugins to more precisely allocate memory for intermediate tensors for inference and use lesser number of tuned kernels for different sizes.
Precisely speaking benefits of specifying upper bound is HW plugin dependent.
Depending on the plugin specifying upper bounds can be required.
<TODO: reference to plugin limitations table>.
If developer knowns lower and upper bounds for dimension it is recommended to specify them even when plugin can execute model without the bounds.

Preparing model with the reshape method was the first step.
The second step is passing a tensor with an appropriate shape to infer request.
This is similar to regular steps described in <TODO: reference to common guide>, but now we can pass tensors with different shapes for the same executable model and even for the same infer request:

@snippet snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:set_input_tensor

In the example above `set_tensor` is used to specify input tensors.
The real dimensions of the tensor is always static, because it is a concrete tensor and it doesn't have any dimension variations in contrast to model inputs.

Similar to static shapes, `InferRequest::get_tensor` can be used instead of `InferRequest::set_tensor`.
In contrast to static input shapes, when using get_tensor for dynamic inputs, `set_sha`pe method of `ov::Tensor` should be called.
Without doing that, the tensor returned by `InferRequest::get_tensor` is an empty tensor, it's shape is not initialized and memory is not allocated, because infer request doesn't have information about real shape.
It is required when input has at least one dynamic dimension regardless of bound information.
The following example makes the same sequence of two infer request as the previous example but using `get_tenso`r instead of `set_tensor` for model input:

@snippet snippets/ov_dynamic_shapes.cpp ov_dynamic_shapes:get_input_tensor

Examples above handle correctly case when dynamic dimensions in output may be implied by propagating of dynamic dimension from the inputs.
For example, batch dimension in input shape is usually propagated through the whole model and appears in the output shape.
The same is true for other dimensions that are propagated through the entire network, not only for batch dimensions.
Whether or not output has a dynamic dimension can be examined by querying output partial shape:

<TODO: Example>

If at least one dynamic dimension exists in output of the model, shape of the corresponding output tensor is set as the result of inference call.
Before the first inference, the output tensor memory is not allocated and has shape `[0]`.
If user call `InferRequest::set_tensor` with pre-allocated tensor, the inference process calls `set_shape` as the result of its work and the initial shape is replaced by the really calculated shape.
So setting shape for output tensors in this case is useful only if you want to pre-allocate enough memory for output tensor, because `Tensor::set_shape` will re-allocate memory only if new shape requires more storage.
