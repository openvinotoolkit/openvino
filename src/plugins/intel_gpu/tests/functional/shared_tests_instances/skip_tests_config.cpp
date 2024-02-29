// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
            // These tests might fail due to accuracy loss a bit bigger than threshold
            R"(.*(GRUCellTest).*)",
            R"(.*(RNNSequenceTest).*)",
            R"(.*(GRUSequenceTest).*)",
            // These test cases might fail due to FP16 overflow
            R"(.*(LSTM).*activations=\(relu.*modelType=f16.*)",

            // Need to update activation primitive to support any broadcastable constant to enable these cases.
            R"(.*ActivationParamLayerTest.*)",
            // Unknown issues
            R"(.*(LSTMSequence).*mode=.*_RAND_SEQ_LEN_CONST.*)",
            R"(.*(smoke_DetectionOutput5In).*)",


            // TODO: Issue: 47773
            R"(.*(ProposalLayerTest).*)",
            // TODO: Issue: 54194
            R"(.*ActivationLayerTest.*SoftPlus.*)",
            R"(.*Behavior.*InferRequestSetBlobByType.*Device=HETERO.*)",
            // TODO: Issue: 59586, NormalizeL2 output mismatch for empty axes case
            R"(.*NormalizeL2LayerTest.*axes=\(\).*)",

            // Not allowed dynamic loop tests on GPU
            R"(.*smoke_StaticShapeLoop_dynamic_exit.*)",
            // TODO Issue 100145
            R"(.*Behavior.*OVInferRequestIOTensorTest.*canInferAfterIOBlobReallocation.*)",
            // Not implemented yet:
            R"(.*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNet.*)",
            // TODO: Issue 67408
            R"(.*smoke_LSTMSequenceCommonClip.*LSTMSequenceTest.*Inference.*)",
            // TODO: Issue 114262
            R"(LSTMSequenceCommonZeroClipNonConstantWRB/LSTMSequenceTest.Inference/mode=PURE_SEQ_seq_lengths=2_batch=10_hidden_size=1_.*relu.*)",
            // Expected behavior. GPU plugin doesn't support i64 for eltwise power operation.
            R"(.*EltwiseLayerTest.*eltwise_op_type=Pow.*model_type=i64.*)",
            // TODO: Issue: 68712
            R"(.*.MatMul.*CompareWithRefs.*IS0=\(1.5\)_IS1=\(1.5\).*transpose_a=0.*transpose_b=1.*CONSTANT.*FP16.*UNSPECIFIED.*UNSPECIFIED.*ANY.*)",
            // Unsupported
            R"(smoke_Behavior/InferRequestSetBlobByType.setInputBlobsByType/BlobType=Batched_Device=GPU_Config=().*)",
            // need dynamic rank
            R"(.*smoke.*BehaviorTests.*InferFullyDynamicNetworkWith(S|G)etTensor.*)",
            R"(.*smoke.*BehaviorTests.*DynamicOutputToDynamicInput.*)",
            R"(.*smoke.*BehaviorTests.*DynamicInputToDynamicOutput.*)",
            // TODO: Issue: 89555
            R"(.*CoreThreadingTestsWithIter.*)",
            // Assign-3/ReadValue-3 does not have evaluate() methods; ref implementation does not save the value across the inferences.
            R"(smoke_MemoryTestV3.*)",
            // Issue: 90539
            R"(smoke_AutoBatch_BehaviorTests/OVInferRequestIOTensorTest.InferStaticNetworkSetInputTensor/targetDevice=BATCH.*)",
            R"(.*CachingSupportCase.*LoadNetworkCacheTestBase.*CompareWithRefImpl.*)",
            // Issue: 119648
            R"(.*smoke_LPT/InterpolateTransformation.*)",
            R"(.*CachingSupportCase.*GPU.*CompileModelCacheTestBase.*CompareWithRefImpl.*)",
            // unsupported metrics
            R"(.*nightly_HeteroAutoBatchOVGetMetricPropsTest.*OVGetMetricPropsTest.*(FULL_DEVICE_NAME_with_DEVICE_ID|AVAILABLE_DEVICES|DEVICE_UUID|OPTIMIZATION_CAPABILITIES|MAX_BATCH_SIZE|DEVICE_GOPS|DEVICE_TYPE|RANGE_FOR_ASYNC_INFER_REQUESTS|RANGE_FOR_STREAMS).*)",
            // Issue: 111437
            R"(.*smoke_Deconv_2D_Dynamic_.*FP32/DeconvolutionLayerGPUTest.Inference.*)",
            R"(.*smoke_GroupDeconv_2D_Dynamic_.*FP32/GroupDeconvolutionLayerGPUTest.Inference.*)",
            // Issue: 111440
            R"(.*smoke_set1/GatherElementsGPUTest.Inference.*)",
            // Issue: Disabled due to LPT precision matching issue
            R"(.*smoke_.*FakeQuantizeTransformation.*)",
            R"(.*smoke_LPT.*ReshapeTransformation.*)",
            R"(.*smoke_LPT.*ConvolutionTransformation.*)",
            R"(.*smoke_LPT.*MatMulWithConstantTransformation.*)",
            R"(.*smoke_LPT.*PullReshapeThroughDequantizationTransformation.*)",
            R"(.*smoke_LPT.*ElementwiseBranchSelectionTransformation.*)",
            // Issue: 123493
            R"(.*GroupNormalizationTest.*CompareWithRefs.*NetType=f16.*)",
            // Issue: 125165
            R"(smoke_Nms9LayerTest.*)",
            // Doesn't match reference results as v6 ref impl behavior is misaligned with expected
            R"(smoke_MemoryTestV3.*)",
            // Issue: 129991
            R"(.*StridedSliceLayerTest.*TS=.*2.2.4.1*.*)",
            // Issue: CVS-133173
            R"(.*smoke_GatherCompressedWeights_basic/GatherWeightsDecompression.Inference/data_shape=\[15,32\]_indices_shape=\[\?.\?\]_\[2.3\].*output_precision=f32.*)",
            R"(.*smoke_CTCLoss_Set2/CTCLossLayerTest.Inference/IS=\(\[\]\)_TS=\{\(3.6.8\)\}_LL=\(6.5.6\)_A=\(4.1.2.3.4.5\)\(5.4.3.0.1.0\)\(2.1.3.1.3.0\)_AL=\(3.3.5\)_BI=7_PCR=1_CMR=1_U=0_PF=f32_PI=i64.*)",
            R"(.*smoke_LPT/BatchToSpaceTransformation.CompareWithRefImpl/f16_GPU_\[4,3,50,86\]_level=256_shape=\[1,1,1,1\]_input_low=\{ 0 \}_input_high=\{ 2.55 \}_output_low=\{ 0 \}_output_high\{ 2.55 \}_precision=.*)",
            R"(.*smoke_LPT/BatchToSpaceTransformation.CompareWithRefImpl/(f32|f16)_GPU_\[4,3,50,86\]_level=256_shape=\[1,3,1,1\]_input_low=\{ 0, 0, 0 \}_input_high=\{ 255, 127.5, 85 \}_output_low=\{ 0, 0, 0 \}_output_high\{ 255, 127.5, 85 \}_precision=.*)",
            R"(.*smoke_LPT/ConcatTransformation.CompareWithRefImpl/f16_\[1,3,16,16\]_GPU_f32level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 2.55 \}_output_low=\{ 0 \}_output_high\{ 2.55 \}_precision=\{\}level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 2.55 \}_output_low=\{ 0 \}_output_high\{ 2.55 \}_precision=\{\}.*)",
            R"(.*smoke_LPT/ConcatWithChildAndOutputTransformation.CompareWithRefImpl/f16_\[1,6,10,10\]_GPU_f32level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 2.55 \}_output_low=\{ 0 \}_output_high\{ 2.55 \}_precision=level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 2.55 \}_output_low=\{ 0 \}_output_high\{ 1.275 \}_precision=.*)",
            R"(.*smoke_LPT/ConcatWithDifferentChildrenTransformation.CompareWithRefImpl/f16_\[1,3,10,10\]_GPU_f32_axis_(1|2)_level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 2.55 \}_output_low=\{ 0 \}_output_high\{ 2.55 \}_precision=level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 2.55 \}_output_low=\{ 0 \}_output_high\{ 1.275 \}_precision=.*)",
            R"(.*smoke_LPT/ConcatWithNeighborsGraphTransformation.CompareWithRefImpl/f16_\[1,3,16,16\]_GPU_f32.*)",
            R"(.*smoke_LPT/ConcatWithIntermediateTransformation.CompareWithRefImpl/f16_\[1,3,16,16\]_GPU_f32.*)",
            R"(.*smoke_LPT/ConcatWithSplitTransformation.CompareWithRefImpl/f16_\[1,6,10,10\]_GPU_f32level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 2.55 \}_output_low=\{ 0 \}_output_high\{ 2.55 \}_precision=_level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 2.55 \}_output_low=\{ 0 \}_output_high\{ 1.275 \}_precision=.*)",
            R"(.*smoke_LPT_4D/ConvolutionBackpropDataTransformation.CompareWithRefImpl/f32_\[1,32,16,16\]_.*_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ 0 \}_output_high\{ 25.5 \}_precision=__0_\[\]_\{  \}_\{  \}___f32_\{\}__\{ 4 \}_f32_\[\]_1_1_undefined.*)",
            R"(.*smoke_LPT_4D/ConvolutionBackpropDataTransformation.CompareWithRefImpl/f16_\[1,(8|32),16,16\]_.*_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ 0 \}_output_high\{ 25.5 \}_precision=__255_\[1,1,1,1\]_\{ 0 \}_\{ 25.4 \}_\{\}.*)",
            R"(.*smoke_LPT_4D/ConvolutionBackpropDataTransformation.CompareWithRefImpl/f16_\[1,(8|32),16,16\]_.*_input_low.*0.*input_high=.*255.*_output_low=.*0.*_output_high.*25.5.*_precision=__0_\[\]_\{  \}_\{  \}___f32_\{\}__\{ 4 \}_f32_\[\]_1_1_undefined.*)",
            R"(.*smoke_LPT_3D/ConvolutionBackpropDataTransformation.CompareWithRefImpl/(f32|f16)_\[1,32,16,16\]_GPU_f32_\[16\]_level=256_shape=\[1,1,1\]_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ 0 \}_output_high\{ 25.5 \}_precision=__0_\[\]_\{  \}_\{  \}___f32_\{\}__\{ 4 \}_f32_\[\]_1_1_undefined.*)",
            R"(.*smoke_LPT/FakeQuantizeAndMaxPoolTransformation.CompareWithRefImpl/f16_\[1,32,72,48\]_GPU_f32.*)",
            R"(.*smoke_LPT/FakeQuantizeAndAvgPoolTransformation.CompareWithRefImpl/f16_\[1,32,72,48\]_GPU_f32.*)",
            R"(.*smoke_LPT/FuseConvertTransformation.CompareWithRefImpl/f32_\[1,4,16,16\]_GPU_f32_.*)",
            R"(.*smoke_LPT/FuseFakeQuantizeAndScaleShiftTransformation.CompareWithRefImpl/f16_GPU_level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 2.55 \}_output_low=\{ 0 \}_output_high\{ 2.55 \}_precision=.*)",
            R"(.*smoke_LPT/MVNTransformation.CompareWithRefImpl/f16_\[1,4,16,16\]_GPU_f32_AxisSet.*)",
            R"(.*smoke_LPT/NormalizeL2Transformation.CompareWithRefImpl/f16_\[1,4,16,16\]_.*)",
            R"(.*smoke_LPT/PadTransformation.CompareWithRefImpl/f16_\[1,3,16,16\]_GPU_f32_level=256_shape=\[1,1,1,1\]_.*_(constant|reflect|symmetric|edge)_.*)",
            R"(.*smoke_LPT/OutputLayersConcat.CompareWithRefImpl/f32_\[1,3,16,16\]_GPU_f32.*)",
            R"(.*smoke_LPT/ReduceMeanTransformation.CompareWithRefImpl/f16_\[1,3,10,10\]_GPU_f32_level=256_shape=\[1,1,1,1\]_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ 0 \}_output_high\{ 127 \}_precision=\{\}\{\}_keepDims__reduce_axis_1_.*)",
            R"(.*smoke_LPT/ReduceMeanTransformation.CompareWithRefImpl/f16_\[1,3,10,10\]_GPU_f32_level=256_shape=\[1,1,1,1\]_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ 0 \}_output_high\{ 127 \}_precision=\{\}\{\}_reduce_axis_1_.*)",
            R"(.*smoke_LPT/ReduceSumTransformation.CompareWithRefImpl/(f32|f16)_\[1,3,10,10\]_GPU_f32_level=256_shape=\[1,1,1,1\]_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ 0 \}_output_high\{ 127 \}_precision=_keepDims__reduce_axis_2_3_.*)",
            R"(.*smoke_LPT/ReduceSumTransformation.CompareWithRefImpl/f16_\[1,3,10,10\]_GPU_f32_level=256_shape=\[1,1,1,1\]_input_low=\{ 2 \}_input_high=\{ 10 \}_output_low=\{ 2 \}_output_high\{ 10 \}_precision=_reduce_axis_2_3_.*)",
            R"(.*smoke_LPT/ReluTransformation.CompareWithRefImpl/f16_GPU_level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 25.5 \}_output_low=\{ 0 \}_output_high\{ 25.5 \}_precision=.*)",
            R"(.*smoke_LPT/ReluTransformation.CompareWithRefImpl/f16_GPU_level=256_shape=\[\]_input_low=\{ 12.75 \}_input_high=\{ 25.5 \}_output_low=\{ 12.75 \}_output_high\{ 25.5 \}_precision=.*)",
            R"(.*smoke_LPT/SpaceToBatchTransformation.CompareWithRefImpl/(f32|f16)_GPU_\[1,3,100,171\]_level=256_shape=\[1,3,1,1\]_input_low=\{ 0, 0, 0 \}_input_high=\{ 255, 127.5, 85 \}_output_low=\{ 0, 0, 0 \}_output_high\{ 255, 127.5, 85 \}_precision=.*)",
            R"(.*smoke_LPT/SpaceToBatchTransformation.CompareWithRefImpl/f16_GPU_\[1,3,100,171\]_level=256_shape=\[1,1,1,1\]_input_low=\{ 0 \}_input_high=\{ 2.55 \}_output_low=\{ 0 \}_output_high\{ 2.55 \}_precision=.*)",
            R"(.*smoke_LPT/SplitTransformation.CompareWithRefImpl/f16_\[1,3,16,16\]_GPU_f32_level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 25.5 \}_output_low=\{ 0 \}_output_high\{ 12.75 \}_precision=_axis=2_n_splits=2.*)",
            R"(.*smoke_LPT/StridedSliceTransformation.CompareWithRefImpl/f16_\[1,3,24,24\]_GPU_f32_.*_precision=_\{ 0, 0, 0, 0 \}_\{ 1, 0, 1, 1 \}_\{ 1, 2, 1, 1 \}_\{ 1, 0, 1, 1 \}_\{ 1, 1, 1, 1 \}.*)",
            R"(.*smoke_LPT/StridedSliceTransformation.CompareWithRefImpl/f16_\[1,3,24,24\]_GPU_f32_.*_precision=_\{ 0, 0, 0, 0 \}_\{ 1, 1, 0, 1 \}_\{ 1, 3, 20, 24 \}_\{ 1, 1, 0, 1 \}_\{ 1, 1, 1, 1 \}.*)",
            R"(.*smoke_LPT/StridedSliceTransformation.CompareWithRefImpl/f16_\[1,3,24,24\]_GPU_f32_level=256_shape=\[1,3,1,1\]_.*_precision=_\{ 0, 0 \}_\{ 1, 0 \}_\{ 1, 2 \}_\{ 1, 0 \}_\{ 1, 1 \}.*)",
            R"(.*smoke_LPT/SubtractTransformation.CompareWithRefImpl/f16_\[1,3,16,16\]_GPU_f32.*)",
            R"(.*smoke_LPT/TransposeAfterMatMulTransformation.CompareWithRefImpl/f16.*(T|t)ransposeChannelDim.*)",
            R"(.*smoke_LPT/VariadicSplitTransformation.CompareWithRefImpl/f16_\[1,3,16,16\]_GPU_f32_level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 25.5 \}_output_low=\{ 0 \}_output_high\{ 12.75 \}_precision=_axis=2_splitLengths=\{ 9, 7 \}.*)",
            R"(.*smoke_ConvolutionBackpropData2D_ExplicitPadding/ConvolutionBackpropDataLayerTest.Inference/IS=\(\[\]\)_TS=.*1.16.10.10.*_OS=\(\)_K\(1.1\)_S\(1.3\).*)",
            R"(.*smoke_ConvolutionBackpropData2D_ExplicitPadding/ConvolutionBackpropDataLayerTest.Inference/IS=\(\[\]\)_TS=.*1.32.10.10.*_OS=\(\)_K\(1.1\)_S\(1.3\).*)",
            R"(.*smoke_ConvolutionBackpropData2D_ExplicitPadding/ConvolutionBackpropDataLayerTest.Inference/IS=\(\[\]\)_TS=.*1.3.30.30.*_OS=\(\)_K\(1.1\)_S\(1.3\).*O=16.*)",
            R"(.*smoke_ConvolutionBackpropData2D_AutoPadValid/ConvolutionBackpropDataLayerTest.Inference/IS=\(\[\]\)_TS=\{\((1\.32\.10\.10|1\.16\.10\.10)\)\}_OS=\(\)_K\(1.1\)_S\(1.3\)_PB\(0.0\)_PE\(0.0\)_D=\(1.1\)_OP=\(\)_O=(1|5|16)_AP=valid_netPRC=f16.*)",
            R"(.*smoke_ConvolutionBackpropData2D_AutoPadValid/ConvolutionBackpropDataLayerTest.Inference/IS=\(\[\]\)_TS=.*1.3.30.30.*_OS=\(\)_K\(1.1\)_S\(1.3\)_PB\(0.0\)_PE\(0.0\)_D=\(1.1\)_OP=\(\)_O=16_AP=valid_netPRC=f16.*)",
            R"(.*smoke_ConvolutionBackpropData2D_ExplicitPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference/IS=\(\[\]\)_TS=\{\((1.32.10.10|1.16.10.10|1.3.30.30)\)\}_OS=\(\)_K\(1.1\)_S\(3.3\)_PB\(0.0\)_PE\(0.0\)_D=\(1.1\)_OP=\((1.1|2.2)\)_O=(1|5|16)_AP=valid_netPRC=f16.*)",
            R"(.*smoke_ConvolutionBackpropData2D_AutoPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference/IS=\(\[\]\)_TS=\{\((1.32.10.10|1.16.10.10|1.3.30.30)\)\}_OS=\(\)_K\(1.1\)_S\(3.3\)_PB\(0.0\)_PE\((0.0|1.1)\)_D=\(1.1\)_OP=\((1.1|2.2)\)_O=(1|5|16).*)",
            R"(.*smoke_GridSample/GridSampleLayerTest.Inference/DS=\((5.2.3.5|5.3.4.6)\)_GS=\((5.7.3.2|5.2.8.2)\)_align_corners=(0|1)_Mode=(bilinear|bicubic)_padding_mode=zeros_model_type=f16_grid_type=f32.*)",
            R"(.*smoke_MatMul_BothTranspose/MatMulLayerTest.Inference/IS=\(\[\]_\[\]\)_TS=\{\(5\)_\(5\)\}_transpose_a=1_transpose_b=1_secondary_input_type=(CONSTANT|PARAMETER)_modelType=(f16|f32).*)",
            R"(.*smoke_dynamic_conv_reshape_fullyconnected/ConvReshapeFullyConnectedDynamicGPUTestDynamic.Inference/IS=\[\?\.64\.1\.\?\.\?\]_\[1\.64\.1\.1\.1\]_model_type=f16.*)",
            R"(.*smoke_empty_tensor/EmptyTensorDynamicGPUTest.Inference/IS=\[\?\]_\[30\]_\[40\]_\[50\]_\[10\]_\[7\]_\[\?.\?\]_\[1.0\]_\[1.8\]_\[1.0\]_\[1.3\]_\[1.20\]_NetType=i32.*)",
    };
}
