# When Dynamic Shapes API is Not Applicable  {#openvino_docs_OV_UG_NoDynamicShapes}

Several approaches to emulate dynamic shapes are considered in this article.
Apply the following methods only if the [native dynamic shape API](ov_dynamic_shapes.md) does not work or does not perform as expected.

## Padding

The model can be designed in a way that supports partially filled tensors.
For the BERT model, use a special input to the model to mask out unused elements.
Therefore, the model can be reshaped for some predefined big sequence length once and compiled once. Then, the input tensors are used only partially with a mask specifying valid tokens.
This approach is called *padding*.

However, padding is not applicable to every model and every use case.
Be aware of the internals of the model to apply padding. Otherwise, if the model is not designed to handle dummy element gracefully in a padding area,
then the results of inference may be totally scrambled, or accuracy is significantly affected.
The model can even crash during inference.

Apart from a bad developer experience,
the main disadvantage of padding is a poor performance due to the time-consuming processing of dummy elements in the padding area,
even if the model is properly designed for padding.
The reason is that such models are usually designed in a way where calculations in the padded area still happen, not affecting the end result.

## Multiple Pre-compiled Models

Another approach to handle arbitrary sized inputs is to pre-compile several models reshaped for different input shapes.
This method works well if the number of different shapes is small enough to afford increased time for multiple reshapes and compilations
as well as increased amount of consumed memory.
As this method cannot be scaled well, it is used in a combination with the padding.
Hence, the model with the most suitable input shape among pre-reshaped models is chosen.
It gives a smaller padding area in comparison to a single model.

## Dimension Partitioning

Another practical but still a complicated approach is to divide the input tensor into multiple chunks along the dynamic dimension.
For example, if there is a batch of independent inputs as a single tensor.
Run multiple inferences, if arbitrary division along batch dimension is possible (for batch dimension it should be possible by the dimension purpose).
Use the approach with several pre-compiled models, choosing sized inputs to have the minimum number of inferences,
having a particular batch size in the input tensor.

For example, if there are models pre-compiled for batch sizes *`1`*, *`2`*, *`4`* and *`8`*,
the input tensor with batch *`5`* can be processed with two inference calls with batch size *`1`* and *`4`*.
(At this point, it is assumed that the batch processing is required for performance reasons. In other cases, just loop over images in a batch
and process image by image with a single compiled model.)
