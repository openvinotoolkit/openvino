// Copyright (C) 2018-2024 Intel Corporation
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
            // TODO: Issue: 145926
            R"(.*CoreThreadingTest.smoke_QueryModel.*)",
            // Assign-3/ReadValue-3 does not have evaluate() methods; ref implementation does not save the value across the inferences.
            R"(smoke_MemoryTestV3.*)",
            // Issue: 90539
            R"(.*CachingSupportCase.*LoadNetworkCacheTestBase.*CompareWithRefImpl.*)",
            // Issue: 119648
            R"(.*smoke_LPT/InterpolateTransformation.*)",
            R"(.*CachingSupportCase.*GPU.*CompileModelCacheTestBase.*CompareWithRefImpl.*)",
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
            // Doesn't match reference results as v6 ref impl behavior is misaligned with expected
            R"(smoke_MemoryTestV3.*)",
            // Issue: CVS-133173
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
            // by calc abs_threshold with expected value
            R"(.*smoke_Convolution2D_ExplicitPadding/ActivatiConvolutionLayerTestonLayerTest.Inference.*netPRC=f16.*)",
            R"(.*smoke_Convolution2D_AutoPadValid/ConvolutionLayerTest.Inference.*netPRC=f16.*)",
            R"(.*smoke_Convolution3D_Basic1/ConvolutionLayerTest.*)",
            R"(.*smoke_ConvolutionBackpropData2D_ExplicitPadding/ConvolutionBackpropDataLayerTest.*)",
            R"(.*smoke_ConvolutionBackpropData2D_AutoPadValid/ConvolutionBackpropDataLayerTest.*K\((3.5|3.3)\).*netPRC=f16.*)",
            R"(.*smoke_ConvolutionBackpropData2D_ExplicitPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.*K\((3.5|3.3)\).*netPRC=f16.*)",
            R"(.*smoke_ConvolutionBackpropData2D_AutoPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference/.*K\((3.5|3.3)\).*PE\(1.1\).*netPRC=f16.*)",
            R"(.*smoke_ConvolutionBackpropData2D_AutoPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference/.*TS=\{\(1.32.10.10\).*K\((3.5|3.3)\).*PE\(0.0\).*netPRC=f16.*)",
            R"(.*smoke_ConvolutionBackpropData2D_AutoPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference/.*TS=\{\((1.3.30.30|1.16.10.10)\).*K\(3.5\).*PE\(0.0\).*netPRC=f16.*)",
            R"(.*smoke_ConvolutionBackpropData2D_AutoPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference/.*TS=\{\((1.3.30.30|1.16.10.10)\).*K\(3.3\).*PE\(0.0\).*O=(1|5|16)_AP=explicit_netPRC=f16.*)",
            R"(.*smoke_ConvolutionBackpropData3D_ExplicitPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference/.*TS=\{\((1.16.5.5.5|1.32.5.5.5)\)\}.*O=(1|5)_AP=valid_netPRC=f16.*)",
            R"(.*smoke_ConvolutionBackpropData3D_ExplicitPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference.*O=16_AP=valid_netPRC=f16.*)",
            R"(.*moke_ConvolutionBackpropData3D_AutoPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference/.*TS=\{\((1.16.5.5.5|1.32.5.5.5)\)\}.*O=(1|5)_AP=valid_netPRC=f16.*)",
            R"(.*moke_ConvolutionBackpropData3D_AutoPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference.*O=16_AP=valid_netPRC=f16.*)",
            R"(.*smoke_DeformableConvolution2D_ExplicitPadding/DeformableConvolutionLayerTest.Inference.*O=(1|5)_AP=explicit_BI_PAD=0_MODULATION=1_netPRC=f16.*)",
            R"(.*smoke_DeformableConvolution2D_AutoPadValid/DeformableConvolutionLayerTest.Inference.*O=(1|5)_AP=valid_BI_PAD=0_MODULATION=1_netPRC=f16.*)",
            R"(.*smoke_DeformableConvolution2D_DeformableGroups_ExplicitPadding/DeformableConvolutionLayerTest.Inference.*O=(1|5)_AP=explicit_BI_PAD=(0|1)_MODULATION=(0|1)_netPRC=f16.*)",
            R"(.*smoke_DeformableConvolution2D_SingleTestCase/DeformableConvolutionLayerTest.Inference.*O=(1|5)_AP=explicit_BI_PAD=(0|1)_MODULATION=(0|1)_netPRC=f16.*)",
            R"(.*smoke_DeformableConvolution2D_MultipleGroup.*/DeformableConvolutionLayerTest.Inference.*O=(1|5)_AP=explicit_BI_PAD=(0|1)_MODULATION=(0|1)_netPRC=f16.*)",
            R"(.*smoke_DFT_5d/DFTLayerTest.Inference/IS=\(\[\]\)_TS=\{\(10.4.8.2.2\)\}_Precision=f32_Axes=\(0.1.2.3\)_signal_size=\(\)_Inverse=0.*)",
            R"(.*smoke_DFT_6d/DFTLayerTest.Inference/IS=\(\[\]\)_TS=\{\(10.4.8.2.5.2\)\}_Precision=f32_Axes=\(0.1.2.3.4\)_signal_.*_Inverse=0.*)",
            R"(.*smoke_ConvolutionLayerGPUTest_ExplicitPad1D/ConvolutionLayerGPUTestDynamic.*netPRC=f16.*)",
            R"(.*smoke_MVN_5D/Mvn6LayerTest.Inference/.*ModelType=f16.*_Ax=\(2.3.4\).*)",
            R"(.*smoke_MVN_5D/Mvn6LayerTest.Inference/.*ModelType=f32.*_Ax=\(2.3.4\).*NormVariance=FALSE.*)",
            R"(.*smoke_MVN_4D/Mvn6LayerTest.Inference/.*TS=\{\(1.10.5.17\)\}.*_ModelType=f16.*Ax=\(2.3\).*)",
            R"(.*smoke_MVN_4D/Mvn6LayerTest.Inference/.*TS=\{\(1.3.8.9\)\}.*_ModelType=f16.*Ax=\((2.3|1.2.3)\).*)",
            R"(.*smoke_MVN_3D/Mvn6LayerTest.Inference/IS=\(\[\]\)_TS=\{\(1.32.17\)\}_ModelType=f16_AxType=(i64|i32)_Ax=\((1.2|2)\).*)",
            R"(.*smoke_MVN_2D/Mvn6LayerTest.Inference.*TS=\{\(2.55\)\}_ModelType=f32_.*)",
            R"(.*smoke_Decomposition_6D/Mvn6LayerTest.Inference.*ModelType=(f16|f32).*Ax=\(0.1.2\).*)",
            R"(.*smoke_Decomposition_6D/Mvn6LayerTest.Inference.*ModelType=(f16|f32).*Ax=\(0.1.5\).*)",
            R"(.*smoke_Decomposition_4D/Mvn6LayerTest.Inference.*ModelType=f16.*Ax=\(1\).*)",
            R"(.*smoke_CTCLoss_Set2/CTCLossLayerTest.Inference/.*_LL=\(6.5.6\)_A=\(2.1.5.3.2.6\)\(3.3.3.3.3.3\)\(6.5.6.5.6.5\)_.*_BI=7_.*_CMR=1_U=1_PF=f16.*)",
            R"(.*smoke_RMSNormDecomposition_basic/RMSNormDecomposition.Inference/.*precision=f32.*)",
            R"(.*smoke_RMSNormDecomposition_basic/RMSNormDecomposition.Inference/IS=\(\[\]_\)_TS=\(\(1.2.6\)\)_input_precision=f16.*)",
            R"(.*smoke_RMSNormDecomposition_basic/RMSNormDecomposition.Inference_cached/IS=\(\[\?.\?.96\]_\)_TS=\(\(1.4.96\)\)_input_precision=f32.*)",
            R"(.*smoke_RMSNormDecomposition_basic/RMSNormDecomposition.Inference_cached/IS=\(\[\?.\?.\?\]_\)_TS=\(\(1.2.16\)\)_input_precision=f32.*)",
            R"(.*smoke_RMSNormDecomposition_basic/RMSNormDecomposition.Inference_cached/IS=\(\[\]_\)_TS=\(\(1.2.6\)\)_input_precision=(f16|f32).*)",
            R"(.*smoke_RMSNormDecomposition_basic/RMSNormDecomposition.Inference_cached/IS=\(\[\]_\)_TS=\(\(1.2.18\)\)_input_precision=f32.*)",
            R"(.*smoke_MM_Static_OneDNN/MatMulLayerGPUTest.Inference.*input_type=PARAMETER_netPRC=f16.*)",
            R"(.*smoke_Decomposition_3D/Mvn6LayerTest.Inference/.*TS=\{\(1.32.17\)\}_ModelType=f16_AxType=.*_Ax=\(0.1.2\).*)",
            R"(.*moke_Decomposition_3D/Mvn6LayerTest.Inference.*TS=\{\(1.37.9\)\}_ModelType=f16_AxType=.*_Ax=\(1\).*)",
            R"(.*smoke_Decomposition_4D/Mvn6LayerTest.Inference/.*TS=\{\(2.19.5.10\)\}_ModelType=f32_AxType=(i32|i64)_Ax=\((0.3|3)\)_NormVariance=FALSE.*)",
            R"(.*smoke_Decomposition_4D/Mvn6LayerTest.Inference/.*TS=\{\(2.19.5.10\)\}_ModelType=f16_AxType=(i32|i64)_Ax=\(0.3\)_NormVariance=TRUE.*)",
            R"(.*smoke_Convolution2D_ExplicitPadding/ConvolutionLayerTest.*netPRC=f16.*)",
            R"(.*smoke_SwiGLUFusion_basic/SwiGLUFusion.Inference.*/IS=\(\[\?.\?.96\]_\)_.*_input_precision=f16.*)",
            R"(.*smoke_dynamic_reduce_deconv_concat/ReduceDeconvConcatDynamicGPUTest.Inference/IS=\[1.32.64.\?.\?\]_\[1.32.64.64.64\]_\[1.8.128.\?.\?.4\]_\[1.8.128.128.128.4\]_model_type=f16.*)",
            R"(.*smoke_GPU_Dynamic/KVCacheTest.Inference.*_precision=f16.*)",
            R"(.*smoke_dynamic_shapeof_activation_sqrt/shapeofActivationDynamicGPUTest.Inference/IS=\[\?.\?.1.64\]_\[1.3136.1.64\]_\[1.49.1.64\]_\[2.49.1.64\]_NetType=f16_targetDevice=GPU_activatioinType=23_inShape=\(\)_constantValue=\(\).*)",
            R"(.*smoke_GroupConvolutionLayerGPUTest_dynamic2D.*/GroupConvolutionLayerGPUTestDynamic.Inference/.*_netPRC=f16.*)",
            R"(.*smoke_(DFT|IDFT|IRDFT)_GPU_4D/DFTLayerGPUTest.CompareWithRefs.*)",
            R"(.*smoke_RDFT_GPU_4D/DFTLayerGPUTest.CompareWithRefs/prec=(f32|f16)_IS0=\[\?.\?.\?.\?\]_TS0=\(\(1.192.36.64\)\)_IS1=\[\?\]_TS1=\(\(1\)\)_IS2=\[\?\]_TS2=\(\(1\)\).*)",
            R"(.*smoke_ConvolutionLayerGPUTest_dynamic.*ConvolutionLayerGPUTestDynamic.*netPRC=f16.*)",
            R"(.*smoke_NoReshape/SplitConvConcat.CompareWithRefImpl/IS=\(1.6.40.40\)_ET=f16_.*)",
            R"(.*smoke_basic/PermConvPermConcat.CompareWithRefs/IS=\(1.1.7.32\)_KS=\(1.3\)_OC=(32|64)_ET=f32.*)",
            R"(.*smoke_basic/PermConvPermConcat.CompareWithRefs/IS=\(1.1.8.16\)_KS=\(1.5\)_OC=(32|64)_ET=f32.*)",
            R"(.*smoke_MAX_and_AVGPool_ValidPad/PoolingLayerTest.Inference.*_AvgPool_ExcludePad=0_K\(3.5\).*modelType=f16.*)",
            R"(.*smoke_MatMul_NoTranspose/MatMulLayerTest.Inference/.*_TS=\{\(1.4.5.6\)_\(1.4.6.4\)\}_.*_input_type=CONSTANT_modelType=f16_.*)",
            R"(.*smoke_MatMul_NoTranspose/MatMulLayerTest.Inference/.*_TS=\{\(4.5.6\)_\(6.3\)\}_.*_input_type=PARAMETER_modelType=f16_.*)",
            R"(.*smoke_MatMul_NoTranspose/MatMulLayerTest.Inference/.*_TS=\{\(9.9.9\)_\(9.9\)\}_.*_input_type=PARAMETER_modelType=f16_.*)",
            R"(.*smoke_MatMul_FirstTranspose/MatMulLayerTest.Inference/.*_TS=\{\(100.65\)_\(100.73\)\}_.*_modelType=f16_.*)",
            R"(.*smoke_MatMul_SecondTranspose/MatMulLayerTest.Inference/.*_TS=\{\(1.16.128\)_\(1.64.128\)\}_.*_modelType=f16_.*)",
            R"(.*smoke_MatMul_SecondTranspose/MatMulLayerTest.Inference/.*_TS=\{\(1.64.80\)_\(1.77.80\)\}_.*_modelType=f16_.*)",
            R"(.*smoke_MatMul_SecondTranspose/MatMulLayerTest.Inference/.*_TS=\{\(65.100\)_\(73.100\)\}_.*_modelType=f16_.*)",
            R"(.*smoke_MatMul_BothTranspose/MatMulLayerTest.Inference/.*_TS=\{\(100.65\)_\(73.100\)\}_.*_modelType=f16_.*)",
            R"(.*smoke_Convolution2D_ExplicitPadding/ConvolutionLayerTest.Inference/.*_TS=\{\(1.3.30.30\)\}_K\(3.5\)_.*_O=5_AP=explicit_netPRC=f16.*)",
            R"(.*smoke_ConvolutionBackpropData3D_AutoPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference/.*_TS=\{\(1.3.10.10.10\)\}_.*_PE\((0.0.0|1.1.1)\)_D=\(1.1.1\)_OP=\((1.1.1|2.2.2)\)_O=16_AP=explicit_netPRC=f16_.*)",
            R"(.*smoke_ConvolutionBackpropData3D_AutoPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference/.*_TS=\{\(1.32.5.5.5\)\}_.*_netPRC=f16_.*)",
            R"(.*smoke_ConvolutionBackpropData3D_AutoPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference/.*_TS=\{\(1.16.5.5.5\)\}_.*_netPRC=f16_.*)",
            R"(.*smoke_ConvolutionBackpropData3D_AutoPadding_OutputPaddingDefined/ConvolutionBackpropDataLayerTest.Inference/.*_TS=\{\(1.16.5.5.5\)\}_.*_netPRC=f16_.*)",
            R"(.*smoke_PSROIPooling_average/PSROIPoolingLayerTest.Inference/IS=\(3.8.16.16\)_coord_shape=\(10.5\)_out_dim=2_group_size=2_scale=(0.625|1)_bins_x=1_bins_y=1_mode=average_modelType=f16.*)",
            R"(.*smoke_RDFT_5d_last_axis/RDFTLayerTest.Inference/IS=\(10.4.8.2.5\)_modelType=f32_Axes=\(0.1.2.3.4\)_SignalSize=\(\).*)",
            // Issue: 136862
            R"(.*smoke_ConditionGPUTest_static/StaticConditionLayerGPUTest.CompareWithRefs/IS=\(3.6\)_netPRC=i8_ifCond=PARAM_targetDevice=GPU_.*)",
            // Issue: 142900
            R"(.*smoke_TestsROIAlign_.*ROIAlignV9LayerTest.*)",

#if defined(_WIN32)
            // by calc abs_threshold with expected value
            R"(.*smoke_RemoteTensor/OVRemoteTensorBatched_Test.NV12toBGR_buffer/(num_batch_4|num_batch_2).*)",
            R"(.*smoke_Check/ConstantResultSubgraphTest.Inference/SubgraphType=SINGLE_COMPONENT_IS=\[1,3,10,10\]_IT=i16_Device=GPU.*)",
#endif
    };
}
