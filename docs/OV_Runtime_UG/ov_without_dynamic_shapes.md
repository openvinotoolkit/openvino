# When Dynamic Shapes API is Not Applicable  {#openvino_docs_OV_UG_NoDynamicShapes}

Several approaches to emulate dynamic shapes are considered in this chapter
Apply these methods only if [native dynamic shape API](ov_dynamic_shapes.md) doesn't work for you or doesn't give desired performance.

## Padding

The model can be designed in a way that supports partially filled tensors.
For the BERT model you can use a special input to the model to mask unused elements out.
So, the model can be reshaped for some predefined big sequence length once and compiled once, and then the input tensors are used only partially with mask specifying valid tokens.
This approach is called *padding*.

However, padding is not applicable to every model and every use case.
You should be aware of model internals to apply padding. Otherwise, if the model is not designed to handle dummy element gracefully in padding area,
then the results of inference may be totally scrambled,
or accuracy is significantly affected.
Model can even crash during inference.

Besides the bad developer experience,
the main disadvantage of padding is a bad performance due to spending time for processing dummy elements in the padding area,
even if the model is properly designed to be used with padding.
It turns out that usually such models are designed in a way where calculations in the padded area still happen not affecting the end result.

## Multiple Precompiled Models

Another approach to handle arbitrary sized inputs is to pre-compile several models reshaped for different input shapes.
This method works well if the number of different shapes is small enough to afford increased time for multiple reshapes and compilations
as well as increased amount of consumed memory.
As this method cannot be scaled well it is used in combination with the padding:
model with the most suitable input shape among pre-reshaped models is chosen.
It gives smaller pad area in comparison to a single model.

## Dimension Partitioning

Another practical but still a complicated approach is when the input tensor can be divided into multiple chunks along the dynamic dimension.
For example, if we have a batch of independent inputs as a single tensor.
If arbitrary division along batch dimension is possible - and for batch dimension it should be possible by the dimension purpose -
you can run multiple inferences using the approach with several pre-compiled models choosing sized to have the minimal number of inferences
having a particular batch size in the input tensor.

For example, if there are models pre-compiled for batch sizes 1, 2, 4 and 8,
the input tensor with batch 5 can be processed with two inference calls with batch size 1 and 4.
(Here it's assumed the batch processing is required for performance reasons, otherwise you can just loop over images in a batch,
and process image by image with a single compiled model.)
