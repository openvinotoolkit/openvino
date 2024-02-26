.. {#../low-precision-transformations_step3_main}

Step 3. Main Transformations
============================


.. meta::
   :description: Learn about main transformations, which are mostly low 
                 precision transformations that handle decomposition and 
                 dequantization operations.

.. toctree::
   :maxdepth: 1
   :hidden:

   AddTransformation <../low-precision-transformations_AddTransformation>
   AvgPoolTransformation <../low-precision-transformations_AvgPoolTransformation>
   BatchToSpaceTransformation <../low-precision-transformations_BatchToSpaceTransformation>
   ClampTransformation <../low-precision-transformations_ClampTransformation>
   ConcatTransformation <../low-precision-transformations_ConcatTransformation>
   ConvolutionTransformation <../low-precision-transformations_ConvolutionTransformation>
   ConvolutionBackpropDataTransformation <../low-precision-transformations_ConvolutionBackpropDataTransformation>
   DepthToSpaceTransformation <../low-precision-transformations_DepthToSpaceTransformation>
   FakeQuantizeDecompositionTransformation <../low-precision-transformations_FakeQuantizeDecompositionTransformation>
   FakeQuantizeTransformation <../low-precision-transformations_FakeQuantizeTransformation>
   InterpolateTransformation <../low-precision-transformations_InterpolateTransformation>
   GroupConvolutionTransformation <../low-precision-transformations_GroupConvolutionTransformation>
   GatherTransformation <../low-precision-transformations_GatherTransformation>
   MatMulTransformation <../low-precision-transformations_MatMulTransformation>
   MaxPoolTransformation <../low-precision-transformations_MaxPoolTransformation>
   MultiplyPartialTransformation <../low-precision-transformations_MultiplyPartialTransformation>
   MVNTransformation <../low-precision-transformations_MVNTransformation>
   NormalizeL2Transformation <../low-precision-transformations_NormalizeL2Transformation>
   PadTransformation<../low-precision-transformations_PadTransformation>
   PReluTransformation <../low-precision-transformations_PReluTransformation>
   ReduceMaxTransformation <../low-precision-transformations_ReduceMaxTransformation>
   ReduceMeanTransformation <../low-precision-transformations_ReduceMeanTransformation>
   ReduceMinTransformation <../low-precision-transformations_ReduceMinTransformation>
   ReduceSumTransformation <../low-precision-transformations_ReduceSumTransformation>
   ReluTransformation <../low-precision-transformations_ReluTransformation>
   ReshapeTransformation <../low-precision-transformations_ReshapeTransformation>
   SpaceToBatchTransformation <../low-precision-transformations_SpaceToBatchTransformation>
   SqueezeTransformation <../low-precision-transformations_SqueezeTransformation>
   ShuffleChannelsTransformation <../low-precision-transformations_ShuffleChannelsTransformation>
   SplitTransformation <../low-precision-transformations_SplitTransformation>
   StridedSliceTransformation <../low-precision-transformations_StridedSliceTransformation>
   TransposeTransformation <../low-precision-transformations_TransposeTransformation>
   UnsqueezeTransformation <../low-precision-transformations_UnsqueezeTransformation>
   VariadicSplitTransformation <../low-precision-transformations_VariadicSplitTransformation>


Main transformations are the majority of low precision transformations. Transformations operate with dequantization operations. Main transformations include:
   
* :doc:`AddTransformation <../low-precision-transformations_AddTransformation>` 
* :doc:`AvgPoolTransformation <../low-precision-transformations_AvgPoolTransformation>` 
* :doc:`BatchToSpaceTransformation <../low-precision-transformations_BatchToSpaceTransformation>` 
* :doc:`ClampTransformation <../low-precision-transformations_ClampTransformation>` 
* :doc:`ConcatTransformation <../low-precision-transformations_ConcatTransformation>` 
* :doc:`ConvolutionTransformation <../low-precision-transformations_ConvolutionTransformation>` 
* :doc:`ConvolutionBackpropDataTransformation <../low-precision-transformations_ConvolutionBackpropDataTransformation>` 
* :doc:`DepthToSpaceTransformation <../low-precision-transformations_DepthToSpaceTransformation>` 
* :doc:`FakeQuantizeDecompositionTransformation <../low-precision-transformations_FakeQuantizeDecompositionTransformation>` 
* :doc:`FakeQuantizeTransformation <../low-precision-transformations_FakeQuantizeTransformation>` 
* :doc:`InterpolateTransformation <../low-precision-transformations_InterpolateTransformation>` 
* :doc:`GroupConvolutionTransformation <../low-precision-transformations_GroupConvolutionTransformation>` 
* :doc:`GatherTransformation <../low-precision-transformations_GatherTransformation>` 
* :doc:`MatMulTransformation <../low-precision-transformations_MatMulTransformation>` 
* :doc:`MaxPoolTransformation <../low-precision-transformations_MaxPoolTransformation>` 
* :doc:`MultiplyPartialTransformation <../low-precision-transformations_MultiplyPartialTransformation>` 
* :doc:`MVNTransformation <../low-precision-transformations_MVNTransformation>` 
* :doc:`NormalizeL2Transformation <../low-precision-transformations_NormalizeL2Transformation>` 
* :doc:`PadTransformation<../low-precision-transformations_PadTransformation>`
* :doc:`PReluTransformation <../low-precision-transformations_PReluTransformation>` 
* :doc:`ReduceMaxTransformation <../low-precision-transformations_ReduceMaxTransformation>` 
* :doc:`ReduceMeanTransformation <../low-precision-transformations_ReduceMeanTransformation>` 
* :doc:`ReduceMinTransformation <../low-precision-transformations_ReduceMinTransformation>` 
* :doc:`ReduceSumTransformation <../low-precision-transformations_ReduceSumTransformation>` 
* :doc:`ReluTransformation <../low-precision-transformations_ReluTransformation>` 
* :doc:`ReshapeTransformation <../low-precision-transformations_ReshapeTransformation>` 
* :doc:`SpaceToBatchTransformation <../low-precision-transformations_SpaceToBatchTransformation>` 
* :doc:`SqueezeTransformation <../low-precision-transformations_SqueezeTransformation>` 
* :doc:`ShuffleChannelsTransformation <../low-precision-transformations_ShuffleChannelsTransformation>` 
* :doc:`SplitTransformation <../low-precision-transformations_SplitTransformation>` 
* :doc:`StridedSliceTransformation <../low-precision-transformations_StridedSliceTransformation>` 
* :doc:`TransposeTransformation <../low-precision-transformations_TransposeTransformation>` 
* :doc:`UnsqueezeTransformation <../low-precision-transformations_UnsqueezeTransformation>` 
* :doc:`VariadicSplitTransformation <../low-precision-transformations_VariadicSplitTransformation>` 

Let's explore some main transformations on the example model. Original model:

.. image:: ../../../../../_static/images/step3_original.svg
   :alt: Original model

Result model after main transformations:

.. image:: ../../../../../_static/images/step3_transformed.svg
   :alt: Transformed model

Changes in the example model after main transformation:

* All ``FakeQuantize`` operations (``fakeQuantize1``, ``fakeQuantize2`` and ``fakeQuantize3``) were decomposed:

  * original ``FakeQuantize`` operations were replaced with new operations with other output intervals and output port precision,
  * dequantization operations.
   
* Dequantization operations were moved via precision preserved (``concat1`` and ``concat2``) and quantized (``convolution2``) operations. 

.. note:: 
   
   The left branch (branch #1) does not require per-tensor quantization. As a result, the ``fakeQuantize1``output interval is [0, 255]. But quantized ``convolution2`` requires per-tensor quantization on the right branch (branch #2). Then all connected ``FakeQuantize`` interval operations (``fakeQuantize1`` and ``fakeQuantize2``) are aligned to have per-tensor quantization after the concatenation (``concat2``) operation.

