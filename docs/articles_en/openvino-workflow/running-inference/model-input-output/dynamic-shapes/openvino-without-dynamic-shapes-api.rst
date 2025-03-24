When Dynamic Shapes API is Not Applicable
=========================================


.. meta::
   :description: The methods to emulate dynamic shapes are applied only if the
                 native dynamic shape API does not work or does not perform
                 as expected.


Several approaches to emulate dynamic shapes are considered in this article.
Apply the following methods only if the :doc:`native dynamic shape API <../dynamic-shapes>` does not work or does not perform as expected.

Padding
####################

The model can be designed in a way that supports partially filled tensors.
For the BERT model, use a special input to the model to mask out unused elements.
Therefore, the model can be reshaped for some predefined big sequence length once and compiled once. Then, the input tensors are used only partially with a mask specifying valid tokens.
This approach is called *padding*.

However, padding is not applicable to every model and every use case.
Be aware of the internals of the model before you apply padding. Otherwise, if the model is not designed to handle dummy elements gracefully in a padding area, the results of inference may be entirely scrambled, or accuracy significantly affected.
The model can even crash during inference.

The main disadvantage of padding, apart from impacting developer experience, is poor performance. Even if the model is properly designed for padding, it is often designed in such a way that the time-consuming processing of dummy elements in the padded area still occurs, not affecting the end result but decreasing inference speed.

Multiple Pre-compiled Models
############################

Another approach to handle arbitrary sized inputs is to pre-compile several models reshaped for different input shapes.
This method works well if the number of different shapes is small enough to afford increased time for multiple reshapes and compilations
as well as increased amount of consumed memory.
As this method cannot be scaled well, it is used in combination with padding.
Hence, the model with the most suitable input shape among pre-reshaped models is chosen.
It gives a smaller padding area in comparison to a single model.

Dimension Partitioning
######################

Another practical but still complicated approach is to divide the input tensor into multiple chunks along the dynamic dimension.
For example, if there is a batch of independent inputs as a single tensor.
If arbitrary division along batch dimension is possible, and it should be possible by the dimension purpose,
run multiple inferences. Use the approach with several pre-compiled models, choosing sized inputs to have the minimum number of inferences,
having a particular batch size in the input tensor.

For example, if there are models pre-compiled for batch sizes ``1``, ``2``, ``4`` and ``8``,
the input tensor with batch ``5`` can be processed with two inference calls with batch size ``1`` and ``4``.
(At this point, it is assumed that the batch processing is required for performance reasons. In other cases, just loop over images in a batch
and process image by image with a single compiled model.)

