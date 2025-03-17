OpenVINO™ Low Precision Transformations
=========================================


.. meta::
   :description: Learn about low precision transformations used to infer a quantized model in low precision with the maximum performance on Intel CPU, GPU, and ARM platforms.

.. toctree::
   :maxdepth: 1
   :caption: Low Precision Transformations
   :hidden:

   Quantization Scheme <low-precision-transformations/quantization-scheme>
   Attributes <low-precision-transformations/lpt-attributes>
   Step 1. Prerequisites transformations <low-precision-transformations/step1-prerequisites>
   Step 2. Markup transformations <low-precision-transformations/step2-markup>
   Step 3. Main transformations <low-precision-transformations/step3-main>
   Step 4. Cleanup transformations <low-precision-transformations/step4-cleanup>


Introduction
############

Low precision transformations (known as LPT) are a set of OpenVINO transformations, which are combined in one library. The library is mandatory part of OpenVINO to infer quantized model in low precision with the maximum performance on Intel CPU, GPU and ARM platforms. The library includes more than 45 transformations and supports more then 30 operations. Some transformations are mandatory, some of them are optional and developed for specific device.

The goal of Low Precision Transformations (LPT) is to transform a quantized model from its original precision (FP16 or FP32) to a low precision (INT8: ``signed int8`` or ``unsigned int8``), so that it is prepared for low precision inference in OpenVINO™ plugin. It is achieved by two main principles:

1. ``FakeQuantize`` operation decomposition to two parts:

   * part 1: quantize operation - new ``FakeQuantize`` operation with output quantization intervals in low precision range (signed int8: [-128, 127] or [-127, 127], unsigned int8: [0, 255] or [0, 256]) and with low precision output (``signed int8`` or ``unsigned int8``).

   * part 2: dequantization operations with low precision input and original precision output.

2. Propagation of the dequantization operation through original model's operations. It is done to avoid dequantization operations before original model operations, thus the quantize operations with low precision output remain before the original model operations.

As result, operation input tensor precisions will be changed from original to low precision and operations can be inferred by OpenVINO™ plugin in low precision.

For a more detailed description on how to quantize a model, see the `Low precision tools <#low-precision-tools>`__ section below. For more information about model quantization, refer to **Brief History of Lower Precision in Deep Learning** section in `this whitepaper <https://www.intel.com/content/dam/develop/external/us/en/documents/lower-numerical-precision-deep-learning-jan2018-754765.pdf>`__.

Input model requirements
########################

LPT transformations propagate dequantization operations through the following operations:

* :doc:`Add-1 <../../../openvino-ir-format/operation-sets/operation-specs/arithmetic/add-1>`
* :doc:`AvgPool-1 <../../../openvino-ir-format/operation-sets/operation-specs/pooling/avg-pool-1>`
* :doc:`Clamp-1 <../../../openvino-ir-format/operation-sets/operation-specs/activation/clamp-1>`
* :doc:`Concat-1 <../../../openvino-ir-format/operation-sets/operation-specs/movement/concat-1>`
* :doc:`Convolution-1 <../../../openvino-ir-format/operation-sets/operation-specs/convolution/convolution-1>`
* :doc:`ConvolutionBackpropData-1 <../../../openvino-ir-format/operation-sets/operation-specs/convolution/convolution-backprop-data-1>`
* :doc:`DepthToSpace-1 <../../../openvino-ir-format/operation-sets/operation-specs/movement/depth-to-space-1>`
* :doc:`FakeQuantize-1 <../../../openvino-ir-format/operation-sets/operation-specs/quantization/fake-quantize-1>`
* :doc:`GroupConvolution-1 <../../../openvino-ir-format/operation-sets/operation-specs/convolution/group-convolution-1>`
* :doc:`Interpolate-1 <../../../openvino-ir-format/operation-sets/operation-specs/image/interpolate-1>`
* :doc:`Interpolate-4 <../../../openvino-ir-format/operation-sets/operation-specs/image/interpolate-4>`
* :doc:`MatMul-1 <../../../openvino-ir-format/operation-sets/operation-specs/matrix/matmul-1>`
* :doc:`MaxPool-1 <../../../openvino-ir-format/operation-sets/operation-specs/pooling/max-pool-1>`
* :doc:`Multiply-1 <../../../openvino-ir-format/operation-sets/operation-specs/arithmetic/multiply-1>`
* :doc:`MVN-1 <../../../openvino-ir-format/operation-sets/operation-specs/normalization/mvn-1>`
* :doc:`NormalizeL2-1 <../../../openvino-ir-format/operation-sets/operation-specs/normalization/normalize-l2-1>`
* :doc:`PRelu-1 <../../../openvino-ir-format/operation-sets/operation-specs/activation/prelu-1>`
* :doc:`ReduceMax-1 <../../../openvino-ir-format/operation-sets/operation-specs/reduction/reduce-max-1>`
* :doc:`ReduceMean-1 <../../../openvino-ir-format/operation-sets/operation-specs/reduction/reduce-mean-1>`
* :doc:`ReduceMin-1 <../../../openvino-ir-format/operation-sets/operation-specs/reduction/reduce-min-1>`
* :doc:`ReduceSum-1 <../../../openvino-ir-format/operation-sets/operation-specs/reduction/reduce-sum-1>`
* :doc:`Relu-1 <../../../openvino-ir-format/operation-sets/operation-specs/activation/relu-1>`
* :doc:`Reshape-1 <../../../openvino-ir-format/operation-sets/operation-specs/shape/reshape-1>`
* :doc:`Split-1 <../../../openvino-ir-format/operation-sets/operation-specs/movement/split-1>`
* :doc:`Squeeze-1 <../../../openvino-ir-format/operation-sets/operation-specs/shape/reshape-1>`
* :doc:`StridedSlice-1 <../../../openvino-ir-format/operation-sets/operation-specs/movement/strided-slice-1>`
* :doc:`Transpose-1 <../../../openvino-ir-format/operation-sets/operation-specs/movement/transpose-1>`
* :doc:`Gather-7 <../../../openvino-ir-format/operation-sets/operation-specs/movement/gather-7>`
* :doc:`Gather-8 <../../../openvino-ir-format/operation-sets/operation-specs/movement/gather-8>`
* :doc:`Unsqueeze-1 <../../../openvino-ir-format/operation-sets/operation-specs/shape/unsqueeze-1>`
* :doc:`VariadicSplit-1 <../../../openvino-ir-format/operation-sets/operation-specs/movement/variadic-split-1>`

If operation is not supported by LPT then dequantization operation will not be propagated, input tensor precisions will not be changed to low precision and operation will be executed in original precision.

For example, if you would like to infer a model with ``Convolution`` operation in low precision then the model can look as on picture below:

.. image:: ../../../../assets/images/model_fq_and_convolution.common.svg
   :alt: Quantized Convolution

There are several supported quantization approaches on activations and on weights. All supported approaches are described in `Quantization approaches <#quantization-approaches>`__ section below. In demonstrated model `FakeQuantize operation quantization <#fakequantize-operation>`__ approach is used.

Low precision tools
+++++++++++++++++++

For more details on how to get a quantized model, refer to :doc:`Model Optimization <../../../../openvino-workflow/model-optimization>` document.

Quantization approaches
#######################

LPT transformations support two quantization approaches:

1. ``FakeQuantize`` operation,
2. Quantize and dequantization operations

Let's explore both approaches in details on ``Convolution`` operation.

FakeQuantize operation
++++++++++++++++++++++

In this case ``FakeQuantize`` operation is used on activations and quantized constant on weights. Original input model:

.. image:: ../../../../assets/images/model_fq_and_convolution.common.svg
   :alt: Original model with FakeQuantize


Quantize and dequantization operations
++++++++++++++++++++++++++++++++++++++

In this case ``FakeQuantize`` operation and ``Convert`` are used as quantize operation and return quantized low precision tensor. After quantize operation on activations there are ``Convert`` and dequantization operations to compensate decomposition. Original input model:

.. image:: ../../../../assets/images/model_qdq_and_convolution.common.svg
   :alt: Original model with Q/DQ

In both cases result is the same. In LPT result model you can see that:

1. if necessary, ``FakeQuantize`` operations on activations were decomposed to two part:

   * new ``FakeQuantize`` operation with updated output intervals in low precision range and low precision output,
   * dequantization operations on activations;

2. if necessary, an existing ``FakeQuantize`` decomposition can be reworked to get better precision;

3. dequantization operations were propagated through ``Convolution``.

LPT result model:

.. image:: ../../../../assets/images/model_fq_and_convolution.transformed.svg
   :alt: Result model

Low precision transformations pipeline
++++++++++++++++++++++++++++++++++++++

LPT transformation pipeline has several steps. For each transformation inside one step pattern matcher is unique per transformation, but each operation can be assigned to several transformations.

.. image:: ../../../../assets/images/low_precision_transformation_pipeline.svg
   :alt: Low precision transformations pipeline

Inside each step LPT transformations handle input model operation by operation, applying transformation matching pattern for each transformation from the step to an operation, and execute transformation if pattern is matched. Decomposition transformation decomposes ``FakeQuantize`` to quantize and dequantization operations. Dequantization operations from previous transformation result is used for the current one and so on, until the end of the model is achieved.

As result, usually all operations are inferred by plugin in low precision. If plugin doesn't support an operation inference in low precision, then corresponding LPT transformation can be disabled, and input tensor precisions for the operation will not be changed. In this case the operation is inferred in the original precision.

Low precision transformations pipeline includes four steps:

* :doc:`Step 1: Prerequisites <low-precision-transformations/step1-prerequisites>`
* :doc:`Step 2: Markup transformations <low-precision-transformations/step2-markup>`
* :doc:`Step 3: Main transformations <low-precision-transformations/step3-main>`
* :doc:`Step 4: Cleanup transformations <low-precision-transformations/step4-cleanup>`

Step 1. Prerequisites
---------------------

This step fuses and propagates some operations in the model to prepare for the next step. It is required for OpenVINO plugins. Transformations:

* :doc:`PullReshapeThroughDequantization <low-precision-transformations/step1-prerequisites/pull-reshape-through-dequantization>`
* :doc:`PullTransposeThroughDequantization <low-precision-transformations/step1-prerequisites/pull-transpose-through-dequantization>`
* :doc:`LinOpSequenceFusion <low-precision-transformations/step1-prerequisites/lin-op-sequence-fusion>`

The model on this step is changed. There are more details in developer guide :doc:`Prerequisites transformations <low-precision-transformations/step1-prerequisites>`.

Step 2. Markup
--------------

This step creates runtime attributes for operations. These attributes will be used in next step. Transformations:

* :doc:`MarkupBias <low-precision-transformations/step2-markup/markup-bias>`
* :doc:`MarkupCanBeQuantized <low-precision-transformations/step2-markup/markup-can-be-quantized>`
* :doc:`MarkupPrecisions <low-precision-transformations/step2-markup/markup-precisions>`
* :doc:`MarkupPerTensorQuantization <low-precision-transformations/step2-markup/markup-per-tensor-quantization>`
* :doc:`MarkupAvgPoolPrecisionPreserved <low-precision-transformations/step2-markup/markup-avg-pool-precision-preserved>`
* :doc:`PropagatePrecisions <low-precision-transformations/step2-markup/propagate-precisions>`
* :doc:`AlignQuantizationIntervals <low-precision-transformations/step2-markup/align-quantization-intervals>`
* :doc:`AlignQuantizationParameters <low-precision-transformations/step2-markup/align-quantization-parameters>`

The model on this step is changed: only new attributes are added to some operations. There are more details in developer guide :doc:`Markup transformations <low-precision-transformations/step2-markup>`.

Step 3. Main transformations, FakeQuantize decomposition and dequantization operations handling
-----------------------------------------------------------------------------------------------

This step has the most transformations. These transformations can be separated in two groups: decomposition transformation and dequantization operations handling. There are more details in developer guide :doc:`Main transformations <low-precision-transformations/step3-main>`.

Transformations:

* :doc:`AddTransformation <low-precision-transformations/step3-main/arithmetic/add>`
* :doc:`AvgPoolTransformation <low-precision-transformations/step3-main/pooling/avg-pool>`
* :doc:`ClampTransformation <low-precision-transformations/step3-main/pooling/avg-pool>`
* :doc:`BatchToSpaceTransformation <low-precision-transformations/step3-main/shape/batch-to-space>`
* :doc:`ConcatTransformation <low-precision-transformations/step3-main/movement/concat>`
* :doc:`ConvolutionTransformation <low-precision-transformations/step3-main/convolution/convolution>`
* :doc:`ConvolutionBackpropDataTransformation <low-precision-transformations/step3-main/convolution/convolution-backprop-data>`
* :doc:`DepthToSpaceTransformation <low-precision-transformations/step3-main/movement/depth-to-space>`
* :doc:`FakeQuantizeDecompositionTransformation <low-precision-transformations/step4-cleanup/fake-quantize-decomposition>`
* :doc:`FakeQuantizeTransformation <low-precision-transformations/step3-main/quantization/fake-quantize>`
* :doc:`InterpolateTransformation <low-precision-transformations/step3-main/image/interpolate>`
* :doc:`GroupConvolutionTransformation <low-precision-transformations/step3-main/convolution/group-convolution>`
* :doc:`GatherTransformation <low-precision-transformations/step3-main/movement/gather>`
* :doc:`MatMulTransformation <low-precision-transformations/step3-main/matrix/mat-mul>`
* :doc:`MaxPoolTransformation <low-precision-transformations/step3-main/pooling/max-pool>`
* :doc:`MultiplyPartialTransformation <low-precision-transformations/step3-main/arithmetic/multiply-partial>`
* :doc:`MVNTransformation <low-precision-transformations/step3-main/normalization/mvn>`
* :doc:`NormalizeL2Transformation <low-precision-transformations/step3-main/normalization/normalize-l2>`
* :doc:`PReluTransformation <low-precision-transformations/step3-main/activation/prelu>`
* :doc:`ReduceMaxTransformation <low-precision-transformations/step3-main/reduction/reduce-max>`
* :doc:`ReduceMeanTransformation <low-precision-transformations/step3-main/reduction/reduce-mean>`
* :doc:`ReduceMinTransformation <low-precision-transformations/step3-main/reduction/reduce-min>`
* :doc:`ReduceSumTransformation <low-precision-transformations/step3-main/reduction/reduce-sum>`
* :doc:`ReluTransformation <low-precision-transformations/step3-main/activation/relu>`
* :doc:`ReshapeTransformation <low-precision-transformations/step3-main/shape/reshape>`
* :doc:`SqueezeTransformation <low-precision-transformations/step3-main/shape/squeeze>`
* :doc:`ShuffleChannelsTransformation <low-precision-transformations/step3-main/movement/shuffle-channels>`
* :doc:`SpaceToBatchTransformation <low-precision-transformations/step3-main/shape/space-to-batch>`
* :doc:`SplitTransformation <low-precision-transformations/step3-main/movement/split>`
* :doc:`StridedSliceTransformation <low-precision-transformations/step3-main/movement/strided-slice>`
* :doc:`TransposeTransformation <low-precision-transformations/step3-main/movement/transpose>`
* :doc:`UnsqueezeTransformation <low-precision-transformations/step3-main/shape/unsqueeze>`
* :doc:`VariadicSplitTransformation <low-precision-transformations/step3-main/movement/variadic-split>`

Decomposition transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decomposition transformations decompose the ``FakeQuantize`` operation to: quantize (``FakeQuantize`` with low precision output) and dequantization operations (opposite to quantize, with low precision input and the original precision output). For dequantization operations LPT uses three operations: ``Convert``, ``Subtract`` and ``Multiply``. Element-wise operations ``Subtract`` and ``Multiply`` have constants on the second branches. If dequantization operations are not handled at the end of LPT pipeline, then they will be fused back to the ``FakeQuantize``.


Original ``FakeQuantize``:

.. image:: ../../../../assets/images/fq.common.svg
   :alt: FakeQuantize operation before LPT

``FakeQuantize`` after decomposition to quantization and dequantization operations:

.. image:: ../../../../assets/images/fq.transformed.svg
   :alt: FakeQuantize operation after LPT

Dequantization operations handling transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this step, LPT transformations fuse dequantization operations or move them through existing model operations as much as possible.

Original ``Convolution`` operation in FP32 with dequantization operations before:

.. image:: ../../../../assets/images/model_fq_and_convolution.common.svg
   :alt: Convolution operation before LPT

``Convolution`` operation in INT8 after decomposition and dequantization operations handling:

.. image:: ../../../../assets/images/model_fq_and_convolution.transformed.svg
   :alt: Convolution operation after LPT


Step 4: Cleanup of the result model
-----------------------------------

LPT cleanup transformations is final stage in LPT pipeline. In this step LPT transformations clean up the result model to avoid not handled dequantization operations: fuse dequantization operations if possible (fuse at least ``Convert`` operations if not` to other model operations to cleanup result model).

Transformations:

* :doc:`EliminateFakeQuantizeTransformation <low-precision-transformations/step4-cleanup/eliminate-fake-quantize>`
* :doc:`FoldConvertTransformation <low-precision-transformations/step4-cleanup/fold-convert>`
* :doc:`FoldFakeQuantizeTransformation <low-precision-transformations/step3-main/quantization/fold-fake-quantize>`
* :doc:`FuseConvertTransformation <low-precision-transformations/step4-cleanup/fuse-convert>`
* :doc:`FuseMultiplyToFakeQuantizeTransformation <low-precision-transformations/step4-cleanup/fuse-multiply-to-fake-quantize>`
* :doc:`FuseSubtractToFakeQuantizeTransformation <low-precision-transformations/step4-cleanup/fuse-subtract-to-fake-quantize>`
* :doc:`MultiplyToGroupConvolutionTransformation <low-precision-transformations/step4-cleanup/multiply-to-group-convolution>`

There are more details in developer guide :doc:`Cleanup transformations <low-precision-transformations/step4-cleanup>`.

``FakeQuantize`` operation with not handled dequantization operations:

.. image:: ../../../../assets/images/fq.transformed.svg
   :alt: TODO: FakeQuantize operation with dequantization operations before LPT

``FakeQuantize`` operation with fused dequantization operations:

.. image:: ../../../../assets/images/fq.common.svg
   :alt: TODO: FakeQuantize operation with fused operations after LPT


Low precision transformations in plugin transformation pipeline
###############################################################

Typical transformation pipeline described below.

Step 1. Common optimizations
++++++++++++++++++++++++++++

This step is optional for LPT but typically is presented in OpenVINO™ plugins. The step doesn't use any LPT transformation. Firstly, the step disables dequantization operations constant folding on constant subgraph on weights to prevent the lost of dequantization info on the next plugin transformations. After that, it optimizes the transformation function and converts operations to operation set 1. Typically, usage of this step is the simplest way to meet LPT requirements for the input quantized model. If plugin can guarantee that LPT input requirements are met, then this step can be skipped.

.. doxygensnippet:: docs/articles_en/assets/snippets/lpt_intel_cpu_plugin.cpp
   :language: cpp
   :fragment: [lpt_common]

Step 2. Low precision transformations execution
+++++++++++++++++++++++++++++++++++++++++++++++
This step is mandatory. It configures and runs LPT transformations.

.. doxygensnippet:: docs/articles_en/assets/snippets/lpt_intel_cpu_plugin.cpp
   :language: cpp
   :fragment: [lpt_execution]

Step 3. Plugin-specific transformations
+++++++++++++++++++++++++++++++++++++++

This step is optional. It modifies the transformation function to a device-specific operation set.

.. doxygensnippet:: docs/articles_en/assets/snippets/lpt_intel_cpu_plugin.cpp
   :language: cpp
   :fragment: [lpt_device]

Result model overview
#####################

Let's explore the resnet-50-tf model, quantized to ``fp16``, which is a TensorFlow
implementation of `ResNet-50 <https://github.com/tensorflow/models/tree/v2.2.0/official/r1/resnet>`__
- an image classification model pre-trained on the ImageNet dataset. Originally
redistributed in the "Saved model" format, converted to a frozen graph using the
"tf.graph_util" module.


Inference
+++++++++

The simplest way to infer the model and collect performance counters is :doc:`Benchmark Application <../../../../get-started/learn-openvino/openvino-samples/benchmark-tool>`.

.. code-block:: sh

   ./benchmark_app -m resnet-50-tf.xml -d CPU -niter 1 -api sync -report_type average_counters  -report_folder pc_report_dir

If you infer the model with the OpenVINO™ CPU plugin and collect performance counters, all operations (except last not quantized SoftMax) are executed in INT8 precision.

Results analysis
++++++++++++++++

Result model depends on different factors:

* The original model quantization possibility and quantization quality. For some models, some operations are not possible to be quantized by NNCF tool. In this case ``FakeQuantize`` operations are absent before these operations and they will be inferred in original precision.
* LPT customization and plugin supported operations. If plugin doesn't support INT8 inference for some operation then corresponding LPT transformation should be disabled and the operation will be inferred in original precision.


Information about layer precision is stored in the performance counters that are
available from the OpenVINO Runtime API. For example, the part of performance counters table for the resnet-50-tf model inferred on CPU Plugin looks as follows:

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

.. doxygensnippet:: docs/articles_en/assets/snippets/lpt_intel_cpu_plugin.cpp
   :language: cpp
   :fragment: [lpt_supported_precisions]

In provided example in result model ``Convolution`` operation inputs must have specific precisions: ``u8`` (unsigned int8) precision on input 0 (on activations) and ``i8`` (signed int8) precision on input 1 (on weights).

Operation per tensor quantization restrictions
++++++++++++++++++++++++++++++++++++++++++++++

This option defines if operation supports per-tensor quantization only. The option value is passed as input argument for ``LowPrecision`` constructor. For example:

.. doxygensnippet:: docs/articles_en/assets/snippets/lpt_intel_cpu_plugin.cpp
   :language: cpp
   :fragment: [per_tensor_quantization]

In provided example in result model ``Convolution`` operations must have per-tensor quantization on input 0 (on activations).

Update precisions
++++++++++++++++++

This option defines if each LPT transformation updates precision or not. The option value is boolean and is passed as ``updatePrecisions`` member of ``LayerTransformation::Params`` which is input argument for ``LowPrecision`` constructor. All transformations are affected. If ``true`` then low precision transformations update precisions to low precision and doesn't if ``false``. Typically this option is used for plugin debugging.

Typical customization use cases
+++++++++++++++++++++++++++++++

Plugin specific customization can be implemented via transformation callbacks. For example: asymmetric quantization support can be easily customizable via ``LayerTransformation::isAsymmetricQuantization`` and ``WeightableLayerTransformation::isAsymmetricOnWeights`` methods usage in callbacks. For example:

.. doxygensnippet:: docs/articles_en/assets/snippets/lpt_intel_cpu_plugin.cpp
   :language: cpp
   :fragment: [asymmetric_quantization]

