// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/core.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

namespace {
bool isGPU1Present() {
    std::string target_device{"GPU"};
    std::string deviceID{"1"};
    ov::Core ie = ov::test::utils::create_core();
    auto deviceIDs = ie.get_property(target_device, ov::available_devices);
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        return false;
    }
    return true;
}

bool immadSupported() {
    ov::Core ie = ov::test::utils::create_core();
    auto properties = ie.get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
    bool support_immad = std::find(properties.begin(), properties.end(), ov::intel_gpu::capability::HW_MATMUL) != properties.end();
    return support_immad;
}
} // namespace

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> returnVal = {
            // These tests might fail due to accuracy loss a bit bigger than threshold
            R"(.*(GRUCellTest).*)",
            R"(.*(RNNSequenceTest).*)",
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
            R"(.*CachingSupportCase.*GPU.*CompileModelCacheTestBase.*CompareWithRefImpl.*)",
            // Issue: 111437
            R"(.*smoke_Deconv_2D_Dynamic_.*FP32/DeconvolutionLayerGPUTest.Inference.*)",
            R"(.*smoke_GroupDeconv_2D_Dynamic_.*FP32/GroupDeconvolutionLayerGPUTest.Inference.*)",
            // Issue: 111440
            R"(.*smoke_set1/GatherElementsGPUTest.Inference.*)",
            // Issue: 168015. Low precision PRelu is not supported on GPU
            R"(.*smoke_LPT.*PReluTransformation.*)",
            // Issue: 168016. Low precision LSTMSequence/GPUSequence are not supported on GPU
            R"(.*smoke_LPT.*RecurrentCellTransformation.*)",
            // Issue: expected precision mismatch
            R"(.*smoke_LPT.*PullReshapeThroughDequantizationTransformation.*)",
            // Issue: accuracy mismatch
            R"(.*smoke_LPT.*FuseDequantizeToFakeQuantizeTransformation.*f32_0_dynamic_\[\]_f32__\{\}_\{\}__\{.0.01.\}_dynamic_\[\]_0_1_dynamic_f32_level=256_shape=\[\]_input_low=\{.0.\}_input_high=\{.2.55.\}_output_low=\{.0.\}_output_high=\{.2.55.\}_output_precision=_constant_precision=)",
            R"(.*smoke_LPT.*MatMulWithConstantTransformation.*\[1,1,3,4\].*level=256_shape=\[1,3,1\]_input_low=\{.0,.0,.0.\}_input_high=\{.25,.24,.25.\}_output_low=\{.0,.0,.0.\}_output_high=\{.25,.24,.25.\}_output_precision=_constant_precision=.*)",
            // Issue: 123493
            R"(.*GroupNormalizationTest.*CompareWithRefs.*NetType=f16.*)",
            // Doesn't match reference results as v6 ref impl behavior is misaligned with expected
            R"(smoke_MemoryTestV3.*)",
            // by calc abs_threshold with expected value
            R"(.*smoke_CTCLoss_Set2/CTCLossLayerTest.Inference/IS=\(\[\]\)_TS=\{\(3.6.8\)\}_LL=\(6.5.6\)_A=\(4.1.2.3.4.5\)\(5.4.3.0.1.0\)\(2.1.3.1.3.0\)_AL=\(3.3.5\)_BI=7_PCR=1_CMR=1_U=0_PF=f32_PI=i64.*)",
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
            R"(.*smoke_RMSNormDecomposition_basic/RMSNormDecomposition.Inference/IS=\(\[\]_\)_TS=\(\(1.2.18\)\)_input_precision=f16.*)",
            R"(.*smoke_RMSNormDecomposition_basic/RMSNormDecomposition.Inference_cached/IS=\(\[\?.\?.96\]_\)_TS=\(\(1.4.96\)\)_input_precision=f32.*)",
            R"(.*smoke_RMSNormDecomposition_basic/RMSNormDecomposition.Inference_cached/IS=\(\[\?.\?.\?\]_\)_TS=\(\(1.2.16\)\)_input_precision=f32.*)",
            R"(.*smoke_RMSNormDecomposition_basic/RMSNormDecomposition.Inference_cached/IS=\(\[\]_\)_TS=\(\(1.2.6\)\)_input_precision=(f16|f32).*)",
            R"(.*smoke_RMSNormDecomposition_basic/RMSNormDecomposition.Inference_cached/IS=\(\[\]_\)_TS=\(\(1.2.18\)\)_input_precision=(f16|f32).*)",
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
            // Use weight from model not from path hint
            R"(.*compile_from_weightless_blob.*)",

#if defined(_WIN32)
            // by calc abs_threshold with expected value
            R"(.*smoke_RemoteTensor/OVRemoteTensorBatched_Test.NV12toBGR_buffer/(num_batch_4|num_batch_2).*)",
            R"(.*smoke_Check/ConstantResultSubgraphTest.Inference/SubgraphType=SINGLE_COMPONENT_IS=\[1,3,10,10\]_IT=i16_Device=GPU.*)",
#endif
    };
    if (!isGPU1Present()) {
        returnVal.push_back(R"(.*nightly_OVClassSpecificDevice0Test/OVSpecificDeviceSetConfigTest.GetConfigSpecificDeviceNoThrow/GPU.1.*)");
        returnVal.push_back(R"(.*nightly_OVClassSpecificDevice0Test/OVSpecificDeviceGetConfigTest.GetConfigSpecificDeviceNoThrow/GPU.1.*)");
        returnVal.push_back(R"(.*nightly_OVClassSpecificDevice0Test/OVSpecificDeviceTestSetConfig.SetConfigSpecificDeviceNoThrow/GPU.1.*)");
        returnVal.push_back(R"(.*nightly_OVClassSetDefaultDeviceIDPropTest/OVClassSetDefaultDeviceIDPropTest.SetDefaultDeviceIDNoThrow/0.*)");
        returnVal.push_back(R"(.*nightly_OVClassSeveralDevicesTest/OVClassSeveralDevicesTestCompileModel.CompileModelActualSeveralDevicesNoThrow/0.*)");
    }
    if (immadSupported()) {
        // Failure list
        // Case (first 20 chars)                                                   Fail Count, Pass Count
        // ----------------------------------------------------------------------  ------------------------
        // LSTMCellCommon/LSTMCellTest.Inference/decomposition0_batch=5_hidden_si  [146, 622]
        // LSTMCellCommon/LSTMCellTest.Inference/decomposition1_batch=5_hidden_si  [132, 636]
        // smoke_MatMulCompressedWeights_corner_cases_basic/MatmulWeightsDecompre  [96, 288]
        // smoke_MaxPool8_ExplicitPad_FloorRounding/MaxPoolingV8LayerTest.Inferen  [64, 128]
        // smoke_MaxPool8_ExplicitPad_CeilRounding/MaxPoolingV8LayerTest.Inferenc  [32, 64]
        // MatMulCompressedWeights_corner_cases_big/MatmulWeightsDecompression.In  [22, 362]
        // smoke_MatMulCompressedWeights_basic/MatmulWeightsDecompression.Inferen  [16, 44]
        // smoke_MatmulAndGatherSharedWeightsDecompression/SharedMatmulAndGatherW  [14, 10]
        // smoke_LoRA_HorizontalFusion/FullyConnectedHorizontalFusion.Inference/d  [12, 0]
        // smoke_LSTMCellCommon/LSTMCellTest.Inference/decomposition0_batch=5_hid  [8, 56]
        // smoke_LSTMCellCommon/LSTMCellTest.Inference/decomposition1_batch=5_hid  [8, 56]
        // smoke_Decomposition_3D/Mvn6LayerTest.Inference/IS=([])_TS={(1.37.9)}_M  [2, 46]
        // smoke_MatmulWeightsDecompressionQuantizeConvolution_basic/MatmulWeight  [2, 10]
        // smoke_MatMulCompressedWeights_dyn_quan/MatmulWeightsDecompression.Infe  [2, 4]
        // smoke_MatMul_NoTranspose/MatMulLayerTest.Inference/IS=([]_[])_TS={(1.2  [2, 2]
        // smoke_static_conv_n_dynamic_concat/ConvStaticConcatDynamicGPUTestDynam  [2, 0]
        // LSTMSequenceCM/LSTMSequenceGPUTest.Inference/mode=PURE_SEQ_seq_lengths  [2, 0]
        // smoke_GroupConvolutionLayerGPUTest_dynamic1DSymPad_Disabled/GroupConvo  [2, 0]
        // LSTMSequenceCommonZeroClip/LSTMSequenceGPUTest.Inference/mode=CONVERT_  [1, 323]
        // LSTMSequenceCommonZeroClip/LSTMSequenceGPUTest.Inference/mode=PURE_SEQ  [1, 323]
        // smoke_ScaledAttnStatic_GPU/ScaledAttnLayerGPUTest.CompareWithRefs/netP  [1, 63]
        // smoke_FC_3D/MatMulLayerGPUTest.Inference/IS=[]_[]_TS=((1.429))_((1.429  [1, 1]
        // Inference_without_convert/BF16WeightsDecompression.Inference_without_c  [1, 1]
        // smoke_ConvolutionLayerGPUTest_3D_tensor_basic/ConvolutionLayerGPUTest.  [1, 0]
        returnVal.push_back(R"(.*smoke_MatMulCompressedWeights_corner_cases_basic/MatmulWeightsDecompre.*)");
        returnVal.push_back(R"(.*smoke_MaxPool8_ExplicitPad_FloorRounding/MaxPoolingV8LayerTest.Inferen.*)");
        returnVal.push_back(R"(.*smoke_MaxPool8_ExplicitPad_CeilRounding/MaxPoolingV8LayerTest.Inferenc.*)");
        returnVal.push_back(R"(.*smoke_MatMulCompressedWeights_basic/MatmulWeightsDecompression.Inferen.*)");
        returnVal.push_back(R"(.*smoke_MatmulAndGatherSharedWeightsDecompression/SharedMatmulAndGatherW.*)");
        returnVal.push_back(R"(.*smoke_LoRA_HorizontalFusion/FullyConnectedHorizontalFusion.Inference/d.*)");
        returnVal.push_back(R"(.*smoke_LSTMCellCommon/LSTMCellTest.Inference/decomposition0_batch=5_hid.*)");
        returnVal.push_back(R"(.*smoke_LSTMCellCommon/LSTMCellTest.Inference/decomposition1_batch=5_hid.*)");
        returnVal.push_back(R"(.*smoke_Decomposition_3D/Mvn6LayerTest.Inference/IS=.*)");
        returnVal.push_back(R"(.*smoke_MatmulWeightsDecompressionQuantizeConvolution_basic/MatmulWeight.*)");
        returnVal.push_back(R"(.*smoke_MatMulCompressedWeights_dyn_quan/MatmulWeightsDecompression.Infe.*)");
        returnVal.push_back(R"(.*smoke_MatMul_NoTranspose/MatMulLayerTest.Inference/IS=.*)");
        returnVal.push_back(R"(.*smoke_GroupConvolutionLayerGPUTest_dynamic1DSymPad_Disabled/GroupConvo.*)");
        returnVal.push_back(R"(.*smoke_static_conv_n_dynamic_concat/ConvStaticConcatDynamicGPUTestDynam.*)");
//        returnVal.push_back(R"(.*smoke_ScaledAttnStatic_GPU/ScaledAttnLayerGPUTest.CompareWithRefs/netP.*)");
        returnVal.push_back(R"(.*smoke_FC_3D/MatMulLayerGPUTest.Inference/.*)");
        returnVal.push_back(R"(.*smoke_ConvolutionLayerGPUTest_3D_tensor_basic/ConvolutionLayerGPUTest..*)");
        returnVal.push_back(R"(.*smoke_MatmulWeightsDecompressionQuantizeConvolution_basic.*)");
        returnVal.push_back(R"(.*smoke_Nms9LayerTest/Nms9LayerTest.Inference/num_batches=2_num_boxes=50.*)");
        returnVal.push_back(R"(.*smoke_ScaledAttnDynamic4D_GPU/ScaledAttnLayerGPUTest.CompareWithRefs/n.*)");
    } else {
        // CVS-172342
        returnVal.push_back(R"(.*smoke_MatMulCompressedWeights_3D_weight.*)");
    }
    return returnVal;
}
