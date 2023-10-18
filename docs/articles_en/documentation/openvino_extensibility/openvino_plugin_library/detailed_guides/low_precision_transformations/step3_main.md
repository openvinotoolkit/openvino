# Step 3. Main Transformations {#openvino_docs_OV_UG_lpt_step3_main}

@sphinxdirective

.. meta::
   :description: Learn about main transformations, which are mostly low 
                 precision transformations that handle decomposition and 
                 dequantization operations.

.. toctree::
   :maxdepth: 1
   :hidden:

   AddTransformation <openvino_docs_OV_UG_lpt_AddTransformation>
   AvgPoolTransformation <openvino_docs_OV_UG_lpt_AvgPoolTransformation>
   BatchToSpaceTransformation <openvino_docs_OV_UG_lpt_BatchToSpaceTransformation>
   ClampTransformation <openvino_docs_OV_UG_lpt_ClampTransformation>
   ConcatTransformation <openvino_docs_OV_UG_lpt_ConcatTransformation>
   ConvolutionTransformation <openvino_docs_OV_UG_lpt_ConvolutionTransformation>
   ConvolutionBackpropDataTransformation <openvino_docs_OV_UG_lpt_ConvolutionBackpropDataTransformation>
   DepthToSpaceTransformation <openvino_docs_OV_UG_lpt_DepthToSpaceTransformation>
   FakeQuantizeDecompositionTransformation <openvino_docs_OV_UG_lpt_FakeQuantizeDecompositionTransformation>
   FakeQuantizeTransformation <openvino_docs_OV_UG_lpt_FakeQuantizeTransformation>
   InterpolateTransformation <openvino_docs_OV_UG_lpt_InterpolateTransformation>
   GroupConvolutionTransformation <openvino_docs_OV_UG_lpt_GroupConvolutionTransformation>
   GatherTransformation <openvino_docs_OV_UG_lpt_GatherTransformation>
   MatMulTransformation <openvino_docs_OV_UG_lpt_MatMulTransformation>
   MaxPoolTransformation <openvino_docs_OV_UG_lpt_MaxPoolTransformation>
   MultiplyPartialTransformation <openvino_docs_OV_UG_lpt_MultiplyPartialTransformation>
   MVNTransformation <openvino_docs_OV_UG_lpt_MVNTransformation>
   NormalizeL2Transformation <openvino_docs_OV_UG_lpt_NormalizeL2Transformation>
   PadTransformation<openvino_docs_OV_UG_lpt_PadTransformation>
   PReluTransformation <openvino_docs_OV_UG_lpt_PReluTransformation>
   ReduceMaxTransformation <openvino_docs_OV_UG_lpt_ReduceMaxTransformation>
   ReduceMeanTransformation <openvino_docs_OV_UG_lpt_ReduceMeanTransformation>
   ReduceMinTransformation <openvino_docs_OV_UG_lpt_ReduceMinTransformation>
   ReduceSumTransformation <openvino_docs_OV_UG_lpt_ReduceSumTransformation>
   ReluTransformation <openvino_docs_OV_UG_lpt_ReluTransformation>
   ReshapeTransformation <openvino_docs_OV_UG_lpt_ReshapeTransformation>
   SpaceToBatchTransformation <openvino_docs_OV_UG_lpt_SpaceToBatchTransformation>
   SqueezeTransformation <openvino_docs_OV_UG_lpt_SqueezeTransformation>
   ShuffleChannelsTransformation <openvino_docs_OV_UG_lpt_ShuffleChannelsTransformation>
   SplitTransformation <openvino_docs_OV_UG_lpt_SplitTransformation>
   StridedSliceTransformation <openvino_docs_OV_UG_lpt_StridedSliceTransformation>
   TransposeTransformation <openvino_docs_OV_UG_lpt_TransposeTransformation>
   UnsqueezeTransformation <openvino_docs_OV_UG_lpt_UnsqueezeTransformation>
   VariadicSplitTransformation <openvino_docs_OV_UG_lpt_VariadicSplitTransformation>


Main transformations are the majority of low precision transformations. Transformations operate with dequantization operations. Main transformations include:
   
* :doc:`AddTransformation <openvino_docs_OV_UG_lpt_AddTransformation>` 
* :doc:`AvgPoolTransformation <openvino_docs_OV_UG_lpt_AvgPoolTransformation>` 
* :doc:`BatchToSpaceTransformation <openvino_docs_OV_UG_lpt_BatchToSpaceTransformation>` 
* :doc:`ClampTransformation <openvino_docs_OV_UG_lpt_ClampTransformation>` 
* :doc:`ConcatTransformation <openvino_docs_OV_UG_lpt_ConcatTransformation>` 
* :doc:`ConvolutionTransformation <openvino_docs_OV_UG_lpt_ConvolutionTransformation>` 
* :doc:`ConvolutionBackpropDataTransformation <openvino_docs_OV_UG_lpt_ConvolutionBackpropDataTransformation>` 
* :doc:`DepthToSpaceTransformation <openvino_docs_OV_UG_lpt_DepthToSpaceTransformation>` 
* :doc:`FakeQuantizeDecompositionTransformation <openvino_docs_OV_UG_lpt_FakeQuantizeDecompositionTransformation>` 
* :doc:`FakeQuantizeTransformation <openvino_docs_OV_UG_lpt_FakeQuantizeTransformation>` 
* :doc:`InterpolateTransformation <openvino_docs_OV_UG_lpt_InterpolateTransformation>` 
* :doc:`GroupConvolutionTransformation <openvino_docs_OV_UG_lpt_GroupConvolutionTransformation>` 
* :doc:`GatherTransformation <openvino_docs_OV_UG_lpt_GatherTransformation>` 
* :doc:`MatMulTransformation <openvino_docs_OV_UG_lpt_MatMulTransformation>` 
* :doc:`MaxPoolTransformation <openvino_docs_OV_UG_lpt_MaxPoolTransformation>` 
* :doc:`MultiplyPartialTransformation <openvino_docs_OV_UG_lpt_MultiplyPartialTransformation>` 
* :doc:`MVNTransformation <openvino_docs_OV_UG_lpt_MVNTransformation>` 
* :doc:`NormalizeL2Transformation <openvino_docs_OV_UG_lpt_NormalizeL2Transformation>` 
* :doc:`PadTransformation<openvino_docs_OV_UG_lpt_PadTransformation>`
* :doc:`PReluTransformation <openvino_docs_OV_UG_lpt_PReluTransformation>` 
* :doc:`ReduceMaxTransformation <openvino_docs_OV_UG_lpt_ReduceMaxTransformation>` 
* :doc:`ReduceMeanTransformation <openvino_docs_OV_UG_lpt_ReduceMeanTransformation>` 
* :doc:`ReduceMinTransformation <openvino_docs_OV_UG_lpt_ReduceMinTransformation>` 
* :doc:`ReduceSumTransformation <openvino_docs_OV_UG_lpt_ReduceSumTransformation>` 
* :doc:`ReluTransformation <openvino_docs_OV_UG_lpt_ReluTransformation>` 
* :doc:`ReshapeTransformation <openvino_docs_OV_UG_lpt_ReshapeTransformation>` 
* :doc:`SpaceToBatchTransformation <openvino_docs_OV_UG_lpt_SpaceToBatchTransformation>` 
* :doc:`SqueezeTransformation <openvino_docs_OV_UG_lpt_SqueezeTransformation>` 
* :doc:`ShuffleChannelsTransformation <openvino_docs_OV_UG_lpt_ShuffleChannelsTransformation>` 
* :doc:`SplitTransformation <openvino_docs_OV_UG_lpt_SplitTransformation>` 
* :doc:`StridedSliceTransformation <openvino_docs_OV_UG_lpt_StridedSliceTransformation>` 
* :doc:`TransposeTransformation <openvino_docs_OV_UG_lpt_TransposeTransformation>` 
* :doc:`UnsqueezeTransformation <openvino_docs_OV_UG_lpt_UnsqueezeTransformation>` 
* :doc:`VariadicSplitTransformation <openvino_docs_OV_UG_lpt_VariadicSplitTransformation>` 

Let's explore some main transformations on the example model. Original model:

.. image:: _static/images/step3_original.svg
   :alt: Original model

Result model after main transformations:

.. image:: _static/images/step3_transformed.svg
   :alt: Transformed model

Changes in the example model after main transformation:

* All ``FakeQuantize`` operations (``fakeQuantize1``, ``fakeQuantize2`` and ``fakeQuantize3``) were decomposed:

  * original ``FakeQuantize`` operations were replaced with new operations with other output intervals and output port precision,
  * dequantization operations.
   
* Dequantization operations were moved via precision preserved (``concat1`` and ``concat2``) and quantized (``convolution2``) operations. 

.. note:: 
   
   The left branch (branch #1) does not require per-tensor quantization. As a result, the ``fakeQuantize1``output interval is [0, 255]. But quantized ``convolution2`` requires per-tensor quantization on the right branch (branch #2). Then all connected ``FakeQuantize`` interval operations (``fakeQuantize1`` and ``fakeQuantize2``) are aligned to have per-tensor quantization after the concatenation (``concat2``) operation.

@endsphinxdirective
