.. {#openvino_docs_OV_UG_lpt}

OpenVINO™ Low Precision Transformations
=========================================


.. meta::
   :description: Learn about low precision transformations used to infer a quantized model in low precision with the maximum performance on Intel CPU, GPU, and ARM platforms.

.. toctree::
   :maxdepth: 1
   :caption: Low Precision Transformations
   :hidden:

   Attributes <openvino_docs_OV_UG_lpt_attributes>
   Step 1. Prerequisites transformations <openvino_docs_OV_UG_lpt_step1_prerequisites>
   Step 2. Markup transformations <openvino_docs_OV_UG_lpt_step2_markup>
   Step 3. Main transformations <openvino_docs_OV_UG_lpt_step3_main>
   Step 4. Cleanup transformations <openvino_docs_OV_UG_lpt_step4_cleanup>


Introduction
############

Low precision transformations (known as LPT) are a set of nGraph transformations, which are combined in one library. The library is mandatory part of OpenVINO to infer quantized model in low precision with the maximum performance on Intel CPU, GPU and ARM platforms. The library includes more than 45 transformations and supports more then 30 operations. Some transformations are mandatory, some of them are optional and developed for specific device.

The goal of Low Precision Transformations (LPT) is to transform a quantized model from its original precision (FP16 or FP32) to a low precision (INT8: ``signed int8`` or ``unsigned int8``), so that it is prepared for low precision inference in OpenVINO™ plugin. It is achieved by two main principles:

1. ``FakeQuantize`` operation decomposition to two parts:  

   * part 1: quantize operation - new ``FakeQuantize`` operation with output quantization intervals in low precision range (signed int8: [-128, 127] or [-127, 127], unsigned int8: [0, 255] or [0, 256]) and with low precision output (``signed int8`` or ``unsigned int8``).

   * part 2: dequantization operations with low precision input and original precision output.

2. Propagation of the dequantization operation through original model's operations. It is done to avoid dequantization operations before original model operations, thus the quantize operations with low precision output remain before the original model operations. 

As result, operation input tensor precisions will be changed from original to low precision and operations can be inferred by OpenVINO™ plugin in low precision.

For a more detailed description on how to quantize a model, see the `Low precision tools <#low-precision-tools>`__ section below. For more information about model quantization, refer to **Brief History of Lower Precision in Deep Learning** section in `this whitepaper <https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training>`__.

Input model requirements
########################

LPT transformations propagate dequantization operations through the following operations:

* :doc:`Add-1 <openvino_docs_ops_arithmetic_Add_1>`
* :doc:`AvgPool-1 <openvino_docs_ops_pooling_AvgPool_1>`
* :doc:`Clamp-1 <openvino_docs_ops_activation_Clamp_1>`
* :doc:`Concat-1 <openvino_docs_ops_movement_Concat_1>`
* :doc:`Convolution-1 <openvino_docs_ops_convolution_Convolution_1>`
* :doc:`ConvolutionBackpropData-1 <openvino_docs_ops_convolution_ConvolutionBackpropData_1>`
* :doc:`DepthToSpace-1 <openvino_docs_ops_movement_DepthToSpace_1>`
* :doc:`FakeQuantize-1 <openvino_docs_ops_quantization_FakeQuantize_1>`
* :doc:`GroupConvolution-1 <openvino_docs_ops_convolution_GroupConvolution_1>`
* :doc:`Interpolate-1 <openvino_docs_ops_image_Interpolate_1>`
* :doc:`Interpolate-4 <openvino_docs_ops_image_Interpolate_4>`
* :doc:`MatMul-1 <openvino_docs_ops_matrix_MatMul_1>`
* :doc:`MaxPool-1 <openvino_docs_ops_pooling_MaxPool_1>`
* :doc:`Multiply-1 <openvino_docs_ops_arithmetic_Multiply_1>`
* :doc:`MVN-1 <openvino_docs_ops_normalization_MVN_1>`
* :doc:`NormalizeL2-1 <openvino_docs_ops_normalization_NormalizeL2_1>`
* :doc:`PRelu-1 <openvino_docs_ops_activation_PReLU_1>`
* :doc:`ReduceMax-1 <openvino_docs_ops_reduction_ReduceMax_1>`
* :doc:`ReduceMean-1 <openvino_docs_ops_reduction_ReduceMean_1>`
* :doc:`ReduceMin-1 <openvino_docs_ops_reduction_ReduceMin_1>`
* :doc:`ReduceSum-1 <openvino_docs_ops_reduction_ReduceSum_1>`
* :doc:`Relu-1 <openvino_docs_ops_activation_ReLU_1>`
* :doc:`Reshape-1 <openvino_docs_ops_shape_Reshape_1>`
* :doc:`Split-1 <openvino_docs_ops_movement_Split_1>`
* :doc:`Squeeze-1 <openvino_docs_ops_shape_Reshape_1>`
* :doc:`StridedSlice-1 <openvino_docs_ops_movement_StridedSlice_1>`
* :doc:`Transpose-1 <openvino_docs_ops_movement_Transpose_1>`
* :doc:`Gather-7 <openvino_docs_ops_movement_Gather_7>`
* :doc:`Gather-8 <openvino_docs_ops_movement_Gather_8>`
* :doc:`Unsqueeze-1 <openvino_docs_ops_shape_Unsqueeze_1>`
* :doc:`VariadicSplit-1 <openvino_docs_ops_movement_VariadicSplit_1>`

If operation is not supported by LPT then dequantization operation will not be propagated, input tensor precisions will not be changed to low precision and operation will be executed in original precision. 

For example, if you would like to infer a model with ``Convolution`` operation in low precision then the model can look as on picture below:

.. image:: _static/images/model_fq_and_convolution.common.svg
   :alt: Quantized Convolution

There are several supported quantization approaches on activations and on weights. All supported approaches are described in `Quantization approaches <#quantization-approaches>`__ section below. In demonstrated model `FakeQuantize operation quantization <#fakequantize-operation>`__ approach is used.

Low precision tools
+++++++++++++++++++

For more details on how to get a quantized model, refer to :doc:`Model Optimization <openvino_docs_model_optimization_guide>` document.

Quantization approaches
#######################

LPT transformations support two quantization approaches:

1. ``FakeQuantize`` operation,
2. Quantize and dequantization operations

Let's explore both approaches in details on ``Convolution`` operation.

FakeQuantize operation
++++++++++++++++++++++ 

In this case ``FakeQuantize`` operation is used on activations and quantized constant on weights. Original input model:  

.. image:: _static/images/model_fq_and_convolution.common.svg
   :alt: Original model with FakeQuantize


Quantize and dequantization operations  
++++++++++++++++++++++++++++++++++++++

In this case ``FakeQuantize`` operation and ``Convert`` are used as quantize operation and return quantized low precision tensor. After quantize operation on activations there are ``Convert`` and dequantization operations to compensate decomposition. Original input model:

.. image:: _static/images/model_qdq_and_convolution.common.svg
   :alt: Original model with Q/DQ

In both cases result is the same. In LPT result model you can see that:

1. if necessary, ``FakeQuantize`` operations on activations were decomposed to two part: 

   * new ``FakeQuantize`` operation with updated output intervals in low precision range and low precision output,
   * dequantization operations on activations;  

2. if necessary, an existing ``FakeQuantize`` decomposition can be reworked to get better precision;  

3. dequantization operations were propagated through ``Convolution``.  

LPT result model:  

.. image:: _static/images/model_fq_and_convolution.transformed.svg
   :alt: Result model

Low precision transformations pipeline
++++++++++++++++++++++++++++++++++++++

LPT transformation pipeline has several steps. For each transformation inside one step pattern matcher is unique per transformation, but each operation can be assigned to several transformations.

.. image:: _static/images/low_precision_transformation_pipeline.svg
   :alt: Low precision transformations pipeline

Inside each step LPT transformations handle input model operation by operation, applying transformation matching pattern for each transformation from the step to an operation, and execute transformation if pattern is matched. Decomposition transformation decomposes ``FakeQuantize`` to quantize and dequantization operations. Dequantization operations from previous transformation result is used for the current one and so on, until the end of the model is achieved.

As result, usually all operations are inferred by plugin in low precision. If plugin doesn't support an operation inference in low precision, then corresponding LPT transformation can be disabled, and input tensor precisions for the operation will not be changed. In this case the operation is inferred in the original precision. 

Low precision transformations pipeline includes four steps:

* :doc:`Step 1: Prerequisites <openvino_docs_OV_UG_lpt_step1_prerequisites>`
* :doc:`Step 2: Markup transformations <openvino_docs_OV_UG_lpt_step2_markup>`
* :doc:`Step 3: Main transformations <openvino_docs_OV_UG_lpt_step3_main>`
* :doc:`Step 4: Cleanup transformations <openvino_docs_OV_UG_lpt_step4_cleanup>`

Step 1. Prerequisites
---------------------

This step fuses and propagates some operations in the model to prepare for the next step. It is required for OpenVINO plugins. Transformations:

* :doc:`PullReshapeThroughDequantization <openvino_docs_OV_UG_lpt_PullReshapeThroughDequantization>`
* :doc:`PullTransposeThroughDequantization <openvino_docs_OV_UG_lpt_PullTransposeThroughDequantization>`
* :doc:`LinOpSequenceFusion <openvino_docs_OV_UG_lpt_LinOpSequenceFusion>`

The model on this step is changed. There are more details in developer guide :doc:`Prerequisites transformations <openvino_docs_OV_UG_lpt_step1_prerequisites>`.

Step 2. Markup
--------------

This step creates runtime attributes for operations. These attributes will be used in next step. Transformations:

* :doc:`MarkupBias <openvino_docs_OV_UG_lpt_MarkupBias>`
* :doc:`MarkupCanBeQuantized <openvino_docs_OV_UG_lpt_MarkupCanBeQuantized>`
* :doc:`MarkupPrecisions <openvino_docs_OV_UG_lpt_MarkupPrecisions>`
* :doc:`MarkupPerTensorQuantization <openvino_docs_OV_UG_lpt_MarkupPerTensorQuantization>`
* :doc:`MarkupAvgPoolPrecisionPreserved <openvino_docs_OV_UG_lpt_MarkupAvgPoolPrecisionPreserved>`
* :doc:`PropagatePrecisions <openvino_docs_OV_UG_lpt_PropagatePrecisions>`
* :doc:`AlignQuantizationIntervals <openvino_docs_OV_UG_lpt_AlignQuantizationIntervals>`
* :doc:`AlignQuantizationParameters <openvino_docs_OV_UG_lpt_AlignQuantizationParameters>`

The model on this step is changed: only new attributes are added to some operations. There are more details in developer guide :doc:`Markup transformations <openvino_docs_OV_UG_lpt_step2_markup>`.

Step 3. Main transformations, FakeQuantize decomposition and dequantization operations handling
-----------------------------------------------------------------------------------------------

This step has the most transformations. These transformations can be separated in two groups: decomposition transformation and dequantization operations handling. There are more details in developer guide :doc:`Main transformations <openvino_docs_OV_UG_lpt_step3_main>`. 

Transformations:

* :doc:`AddTransformation <openvino_docs_OV_UG_lpt_AddTransformation>`
* :doc:`AvgPoolTransformation <openvino_docs_OV_UG_lpt_AvgPoolTransformation>`
* :doc:`ClampTransformation <openvino_docs_OV_UG_lpt_AvgPoolTransformation>`
* :doc:`BatchToSpaceTransformation <openvino_docs_OV_UG_lpt_BatchToSpaceTransformation>`
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
* :doc:`PReluTransformation <openvino_docs_OV_UG_lpt_PReluTransformation>`
* :doc:`ReduceMaxTransformation <openvino_docs_OV_UG_lpt_ReduceMaxTransformation>`
* :doc:`ReduceMeanTransformation <openvino_docs_OV_UG_lpt_ReduceMeanTransformation>`
* :doc:`ReduceMinTransformation <openvino_docs_OV_UG_lpt_ReduceMinTransformation>`
* :doc:`ReduceSumTransformation <openvino_docs_OV_UG_lpt_ReduceSumTransformation>`
* :doc:`ReluTransformation <openvino_docs_OV_UG_lpt_ReluTransformation>`
* :doc:`ReshapeTransformation <openvino_docs_OV_UG_lpt_ReshapeTransformation>`
* :doc:`SqueezeTransformation <openvino_docs_OV_UG_lpt_SqueezeTransformation>`
* :doc:`ShuffleChannelsTransformation <openvino_docs_OV_UG_lpt_ShuffleChannelsTransformation>`
* :doc:`SpaceToBatchTransformation <openvino_docs_OV_UG_lpt_SpaceToBatchTransformation>`
* :doc:`SplitTransformation <openvino_docs_OV_UG_lpt_SplitTransformation>`
* :doc:`StridedSliceTransformation <openvino_docs_OV_UG_lpt_StridedSliceTransformation>`
* :doc:`TransposeTransformation <openvino_docs_OV_UG_lpt_TransposeTransformation>`
* :doc:`UnsqueezeTransformation <openvino_docs_OV_UG_lpt_UnsqueezeTransformation>`
* :doc:`VariadicSplitTransformation <openvino_docs_OV_UG_lpt_VariadicSplitTransformation>`

Decomposition transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decomposition transformations decompose the ``FakeQuantize`` operation to: quantize (``FakeQuantize`` with low precision output) and dequantization operations (opposite to quantize, with low precision input and the original precision output). For dequantization operations LPT uses three operations: ``Convert``, ``Subtract`` and ``Multiply``. Element-wise operations ``Subtract`` and ``Multiply`` have constants on the second branches. If dequantization operations are not handled at the end of LPT pipeline, then they will be fused back to the ``FakeQuantize``.


Original ``FakeQuantize``:  

.. image:: _static/images/fq.common.svg
   :alt: FakeQuantize operation before LPT

``FakeQuantize`` after decomposition to quantization and dequantization operations:   

.. image:: _static/images/fq.transformed.svg
   :alt: FakeQuantize operation after LPT

Dequantization operations handling transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this step, LPT transformations fuse dequantization operations or move them through existing model operations as much as possible.

Original ``Convolution`` operation in FP32 with dequantization operations before:  

.. image:: _static/images/model_fq_and_convolution.common.svg
   :alt: Convolution operation before LPT

``Convolution`` operation in INT8 after decomposition and dequantization operations handling:   

.. image:: _static/images/model_fq_and_convolution.transformed.svg
   :alt: Convolution operation after LPT


Step 4: Cleanup of the result model
-----------------------------------

LPT cleanup transformations is final stage in LPT pipeline. In this step LPT transformations clean up the result model to avoid not handled dequantization operations: fuse dequantization operations if possible (fuse at least ``Convert`` operations if not` to other model operations to cleanup result model). 

Transformations:

* :doc:`EliminateFakeQuantizeTransformation <openvino_docs_OV_UG_lpt_EliminateFakeQuantizeTransformation>`
* :doc:`FoldConvertTransformation <openvino_docs_OV_UG_lpt_FoldConvertTransformation>`
* :doc:`FoldFakeQuantizeTransformation <openvino_docs_OV_UG_lpt_FoldFakeQuantizeTransformation>`
* :doc:`FuseConvertTransformation <openvino_docs_OV_UG_lpt_FuseConvertTransformation>`
* :doc:`FuseMultiplyToFakeQuantizeTransformation <openvino_docs_OV_UG_lpt_FuseMultiplyToFakeQuantizeTransformation>`
* :doc:`FuseSubtractToFakeQuantizeTransformation <openvino_docs_OV_UG_lpt_FuseSubtractToFakeQuantizeTransformation>`
* :doc:`MultiplyToGroupConvolutionTransformation <openvino_docs_OV_UG_lpt_MultiplyToGroupConvolutionTransformation>`

There are more details in developer guide :doc:`Cleanup transformations <openvino_docs_OV_UG_lpt_step4_cleanup>`.

``FakeQuantize`` operation with not handled dequantization operations:  

.. image:: _static/images/fq.transformed.svg
   :alt: TODO: FakeQuantize operation with dequantization operations before LPT

``FakeQuantize`` operation with fused dequantization operations:  

.. image:: _static/images/fq.common.svg
   :alt: TODO: FakeQuantize operation with fused operations after LPT


Low precision transformations in plugin transformation pipeline
###############################################################

Typical transformation pipeline described below.

Step 1. Common optimizations
++++++++++++++++++++++++++++

This step is optional for LPT but typically is presented in OpenVINO™ plugins. The step doesn't use any LPT transformation. Firstly, the step disables dequantization operations constant folding on constant subgraph on weights to prevent the lost of dequantization info on the next plugin transformations. After that, it optimizes nGraph function and convert operations to operation set 1. Typically, usage of this step is the simplest way to meet LPT requirements for the input quantized model. If plugin can guarantee that LPT input requirements are met, then this step can be skipped.

.. doxygensnippet:: docs/snippets/lpt_intel_cpu_plugin.cpp
   :language: cpp
   :fragment: [lpt_common]

Step 2. Low precision transformations execution  
+++++++++++++++++++++++++++++++++++++++++++++++
This step is mandatory. It configures and runs LPT transformations.

.. doxygensnippet:: docs/snippets/lpt_intel_cpu_plugin.cpp
   :language: cpp
   :fragment: [lpt_execution]

Step 3. Plugin-specific transformations  
+++++++++++++++++++++++++++++++++++++++

This step is optional. It modifies the nGraph function to a device-specific operation set.

.. doxygensnippet:: docs/snippets/lpt_intel_cpu_plugin.cpp
   :language: cpp
   :fragment: [lpt_device]

Result model overview
#####################

Let's explore quantized `TensorFlow implementation of ResNet-50 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf>`__ model. Use :doc:`Model Downloader <omz_tools_downloader>` tool to download the ``fp16`` model from `OpenVINO™ Toolkit - Open Model Zoo repository <https://github.com/openvinotoolkit/open_model_zoo>`__:

.. code-block:: sh

   omz_downloader --name resnet-50-tf --precisions FP16-INT8

After that you should quantize model by the :doc:`Model Quantizer <omz_tools_downloader>` tool.

.. code-block:: sh

   omz_quantizer --model_dir public/resnet-50-tf --dataset_dir <DATASET_DIR> --precisions=FP16-INT8


Inference
+++++++++

The simplest way to infer the model and collect performance counters is :doc:`Benchmark Application <openvino_inference_engine_samples_benchmark_app_README>`.

.. code-block:: sh 

   ./benchmark_app -m resnet-50-tf.xml -d CPU -niter 1 -api sync -report_type average_counters  -report_folder pc_report_dir

If you infer the model with the OpenVINO™ CPU plugin and collect performance counters, all operations (except last not quantized SoftMax) are executed in INT8 precision.  

Results analysis
++++++++++++++++

Result model depends on different factors:

* The original model quantization possibility and quantization quality. For some models, some operations are not possible to be quantized by NNCF tool. In this case ``FakeQuantize`` operations are absent before these operations and they will be inferred in original precision.
* LPT customization and plugin supported operations. If plugin doesn't support INT8 inference for some operation then corresponding LPT transformation should be disabled and the operation will be inferred in original precision.


Information about layer precision is stored in the performance counters that are
available from the OpenVINO Runtime API. For example, the part of performance counters table for quantized `TensorFlow implementation of ResNet-50 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf>`__  model inference on CPU Plugin looks as follows:

.. list-table::
    :header-rows: 1

    * - layerName
      - execStatus
      - layerType
      - execType
      - realTime (ms)
      - cpuTime (ms)
    * - resnet_model/batch_normalization_15/FusedBatchNorm/Add
      - EXECUTED
      - Convolution
      - jit_avx512_1x1_I8
      - 0.377
      - 0.377
    * - resnet_model/conv2d_16/Conv2D/fq_input_0
      - NOT_RUN
      - FakeQuantize
      - undef
      - 0
      - 0
    * - resnet_model/batch_normalization_16/FusedBatchNorm/Add
      - EXECUTED
      - Convolution
      - jit_avx512_I8
      - 0.499
      - 0.499
    * - resnet_model/conv2d_17/Conv2D/fq_input_0
      - NOT_RUN
      - FakeQuantize
      - undef
      - 0
      - 0
    * - resnet_model/batch_normalization_17/FusedBatchNorm/Add
      - EXECUTED
      - Convolution
      - jit_avx512_1x1_I8
      - 0.399
      - 0.399
    * - resnet_model/add_4/fq_input_0
      - NOT_RUN
      - FakeQuantize
      - undef
      - 0
      - 0
    * - resnet_model/add_4
      - NOT_RUN
      - Eltwise
      - undef
      - 0
      - 0
    * - resnet_model/add_5/fq_input_1
      - NOT_RUN
      - FakeQuantize
      - undef
      - 0
      - 0


The ``execStatus`` column of the table includes possible values:

* ``EXECUTED`` - layer was executed by standalone primitive,
* ``NOT_RUN`` - layer was not executed by standalone primitive or was fused with another operation and executed in another layer primitive.  

The ``execType`` column of the table includes inference primitives with specific suffixes. The layers have the following marks:

* Suffix ``I8`` for layers that had 8-bit data type input and were computed in 8-bit precision
* Suffix ``FP32`` for layers computed in 32-bit precision 

As result all operations (except not quantized ``SoftMax`` at the end of the model) in OpenVINO™ CPU plugin are inferred in low precision. Note, please, in the result model there are ``FakeQuantize`` operations in FP32 but the plugin responsibility is fuse these operations with previous operations. OpenVINO™ CPU plugin achieves maximum optimized inference for all operations by fusing INT8 ``Convolution`` with FP32 output with ``FakeQuantize`` operation with FP32 input and INT8 output. In this case OpenVINO™ CPU plugin uses INT8 and FP32 vectorized instructions but reports about one INT8 kernel usage for inference, which is the most optimized for this case.

Mixed precision
###############

If LPT input model operation output has ``fp16`` precision then dequantization computations still occurs in ``fp32`` precision. This approach is used to avoid accuracy loss in ``fp16`` arithmetic computations. The ultimate output of the dequantization operation  will have the ``fp16`` precision, as expected.

Customization
#############

Low Precision Transformations can be customizable. Build-in customization options:

* operation precision restrictions,
* operation per tensor quantization restrictions,
* update precisions,
* dequantization precision.

Operation precision restrictions
++++++++++++++++++++++++++++++++

This option defines precisions which allowed for the operation input ports. The option value is passed as input argument for ``LowPrecision`` constructor. For example:

.. doxygensnippet:: docs/snippets/lpt_intel_cpu_plugin.cpp
   :language: cpp
   :fragment: [lpt_supported_precisions]
   
In provided example in result model ``Convolution`` operation inputs must have specific precisions: ``u8`` (unsigned int8) precision on input 0 (on activations) and ``i8`` (signed int8) precision on input 1 (on weights).

Operation per tensor quantization restrictions
++++++++++++++++++++++++++++++++++++++++++++++

This option defines if operation supports per-tensor quantization only. The option value is passed as input argument for ``LowPrecision`` constructor. For example:

.. doxygensnippet:: docs/snippets/lpt_intel_cpu_plugin.cpp
   :language: cpp
   :fragment: [per_tensor_quantization]

In provided example in result model ``Convolution`` operations must have per-tensor quantization on input 0 (on activations).

Update precisions
++++++++++++++++++

This option defines if each LPT transformation updates precision or not. The option value is boolean and is passed as ``updatePrecisions`` member of ``LayerTransformation::Params`` which is input argument for ``LowPrecision`` constructor. All transformations are affected. If ``true`` then low precision transformations update precisions to low precision and doesn't if ``false``. Typically this option is used for plugin debugging.

Typical customization use cases
+++++++++++++++++++++++++++++++

Plugin specific customization can be implemented via nGraph transformation callbacks. For example: asymmetric quantization support can be easily customizable via ``LayerTransformation::isAsymmetricQuantization`` and ``WeightableLayerTransformation::isAsymmetricOnWeights`` methods usage in callbacks. For example:

.. doxygensnippet:: docs/snippets/lpt_intel_cpu_plugin.cpp
   :language: cpp
   :fragment: [asymmetric_quantization]

