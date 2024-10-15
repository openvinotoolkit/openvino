Step 3. Main Transformations
============================


.. meta::
   :description: Learn about main transformations, which are mostly low
                 precision transformations that handle decomposition and
                 dequantization operations.

.. toctree::
   :maxdepth: 1
   :hidden:

   AddTransformation <step3-main/arithmetic/add>
   AvgPoolTransformation <step3-main/pooling/avg-pool>
   BatchToSpaceTransformation <step3-main/shape/batch-to-space>
   ClampTransformation <step3-main/activation/clamp>
   ConcatTransformation <step3-main/movement/concat>
   ConvolutionTransformation <step3-main/convolution/convolution>
   ConvolutionBackpropDataTransformation <step3-main/convolution/convolution-backprop-data>
   DepthToSpaceTransformation <step3-main/movement/depth-to-space>
   FakeQuantizeDecompositionTransformation <step4-cleanup/fake-quantize-decomposition>
   FakeQuantizeTransformation <step3-main/quantization/fake-quantize>
   InterpolateTransformation <step3-main/image/interpolate>
   GroupConvolutionTransformation <step3-main/convolution/group-convolution>
   GatherTransformation <step3-main/movement/gather>
   MatMulTransformation <step3-main/matrix/mat-mul>
   MaxPoolTransformation <step3-main/pooling/max-pool>
   MultiplyPartialTransformation <step3-main/arithmetic/multiply-partial>
   MultiplyTransformation <step3-main/arithmetic/multiply>
   MVNTransformation <step3-main/normalization/mvn>
   NormalizeL2Transformation <step3-main/normalization/normalize-l2>
   PadTransformation<step3-main/movement/pad>
   PReluTransformation <step3-main/activation/prelu>
   ReduceMaxTransformation <step3-main/reduction/reduce-max>
   ReduceMeanTransformation <step3-main/reduction/reduce-mean>
   ReduceMinTransformation <step3-main/reduction/reduce-min>
   ReduceSumTransformation <step3-main/reduction/reduce-sum>
   ReluTransformation <step3-main/activation/relu>
   ReshapeTransformation <step3-main/shape/reshape>
   SpaceToBatchTransformation <step3-main/shape/space-to-batch>
   SqueezeTransformation <step3-main/shape/squeeze>
   ShuffleChannelsTransformation <step3-main/movement/shuffle-channels>
   SplitTransformation <step3-main/movement/split>
   StridedSliceTransformation <step3-main/movement/strided-slice>
   SubtractTransformation <step3-main/arithmetic/subtract>
   TransposeTransformation <step3-main/movement/transpose>
   UnsqueezeTransformation <step3-main/shape/unsqueeze>
   VariadicSplitTransformation <step3-main/movement/variadic-split>


Main transformations are the majority of low precision transformations. Transformations operate with dequantization operations. Main transformations include:

* :doc:`AddTransformation <step3-main/arithmetic/add>`
* :doc:`AvgPoolTransformation <step3-main/pooling/avg-pool>`
* :doc:`BatchToSpaceTransformation <step3-main/shape/batch-to-space>`
* :doc:`ClampTransformation <step3-main/activation/clamp>`
* :doc:`ConcatTransformation <step3-main/movement/concat>`
* :doc:`ConvolutionTransformation <step3-main/convolution/convolution>`
* :doc:`ConvolutionBackpropDataTransformation <step3-main/convolution/convolution-backprop-data>`
* :doc:`DepthToSpaceTransformation <step3-main/movement/depth-to-space>`
* :doc:`FakeQuantizeDecompositionTransformation <step4-cleanup/fake-quantize-decomposition>`
* :doc:`FakeQuantizeTransformation <step3-main/quantization/fake-quantize>`
* :doc:`InterpolateTransformation <step3-main/image/interpolate>`
* :doc:`GroupConvolutionTransformation <step3-main/convolution/group-convolution>`
* :doc:`GatherTransformation <step3-main/movement/gather>`
* :doc:`MatMulTransformation <step3-main/matrix/mat-mul>`
* :doc:`MaxPoolTransformation <step3-main/pooling/max-pool>`
* :doc:`MultiplyPartialTransformation <step3-main/arithmetic/multiply-partial>`
* :doc:`MultiplyTransformation <step3-main/arithmetic/multiply>`
* :doc:`MVNTransformation <step3-main/normalization/mvn>`
* :doc:`NormalizeL2Transformation <step3-main/normalization/normalize-l2>`
* :doc:`PadTransformation <step3-main/movement/pad>`
* :doc:`PReluTransformation <step3-main/activation/prelu>`
* :doc:`ReduceMaxTransformation <step3-main/reduction/reduce-max>`
* :doc:`ReduceMeanTransformation <step3-main/reduction/reduce-mean>`
* :doc:`ReduceMinTransformation <step3-main/reduction/reduce-min>`
* :doc:`ReduceSumTransformation <step3-main/reduction/reduce-sum>`
* :doc:`ReluTransformation <step3-main/activation/relu>`
* :doc:`ReshapeTransformation <step3-main/shape/reshape>`
* :doc:`SpaceToBatchTransformation <step3-main/shape/space-to-batch>`
* :doc:`SqueezeTransformation <step3-main/shape/squeeze>`
* :doc:`ShuffleChannelsTransformation <step3-main/movement/shuffle-channels>`
* :doc:`SplitTransformation <step3-main/movement/split>`
* :doc:`StridedSliceTransformation <step3-main/movement/strided-slice>`
* :doc:`SubtractTransformation <step3-main/arithmetic/subtract>`
* :doc:`TransposeTransformation <step3-main/movement/transpose>`
* :doc:`UnsqueezeTransformation <step3-main/shape/unsqueeze>`
* :doc:`VariadicSplitTransformation <step3-main/movement/variadic-split>`

Let's explore some main transformations on the example model. Original model:

.. image:: ../../../../../assets/images/step3_original.svg
   :alt: Original model

Result model after main transformations:

.. image:: ../../../../../assets/images/step3_transformed.svg
   :alt: Transformed model

Changes in the example model after main transformation:

* All ``FakeQuantize`` operations (``fakeQuantize1``, ``fakeQuantize2`` and ``fakeQuantize3``) were decomposed:

  * original ``FakeQuantize`` operations were replaced with new operations with other output intervals and output port precision,
  * dequantization operations.

* Dequantization operations were moved via precision preserved (``concat1`` and ``concat2``) and quantized (``convolution2``) operations.

.. note::

   The left branch (branch #1) does not require per-tensor quantization. As a result, the ``fakeQuantize1`` output interval is [0, 255]. But quantized ``convolution2`` requires per-tensor quantization on the right branch (branch #2). Then all connected ``FakeQuantize`` interval operations (``fakeQuantize1`` and ``fakeQuantize2``) are aligned to have per-tensor quantization after the concatenation (``concat2``) operation.

