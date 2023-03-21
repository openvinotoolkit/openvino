// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
            // TODO: Issue: 39612
            R"(.*Interpolate.*cubic.*tf_half_pixel_for_nn.*FP16.*)",
            // TODO: Issue: 43794
            R"(.*(PreprocessTest).*(SetScalePreProcessSetBlob).*)",
            R"(.*(PreprocessTest).*(SetScalePreProcessGetBlob).*)",
            R"(.*(PreprocessTest).*(SetMeanValuePreProcessSetBlob).*)",
            R"(.*(PreprocessTest).*(SetMeanImagePreProcessSetBlob).*)",
            R"(.*(PreprocessTest).*(ReverseInputChannelsPreProcessGetBlob).*)",
            R"(.*(InferRequestPreprocessDynamicallyInSetBlobTest).*)",
            // TODO: Issue: 46841
            R"(.*(QuantGroupConvBackpropData3D).*)",

            // These tests might fail due to accuracy loss a bit bigger than threshold
            R"(.*(GRUCellTest).*)",
            R"(.*(RNNSequenceTest).*)",
            R"(.*(GRUSequenceTest).*)",
            // These test cases might fail due to FP16 overflow
            R"(.*(LSTM).*activations=\(relu.*netPRC=FP16.*)",

            // Need to update activation primitive to support any broadcastable constant to enable these cases.
            R"(.*ActivationParamLayerTest.*)",
            // Unknown issues
            R"(.*(LSTMSequence).*mode=.*_RAND_SEQ_LEN_CONST.*)",
            R"(.*(smoke_DetectionOutput5In).*)",
            // TODO: Issue: 47773
            R"(.*(ProposalLayerTest).*)",
            // TODO: Issue: 48106
            R"(.*ConstantResultSubgraphTest.*inPrc=I16.*)",
            // TODO: Issue: 54194
            R"(.*ActivationLayerTest.*SoftPlus.*)",
            // need to implement Export / Import
            R"(.*IEClassImportExportTestP.*)",
            R"(.*Behavior.*InferRequestSetBlobByType.*Device=HETERO.*)",
            // TODO: Issue: 59586, NormalizeL2 output mismatch for empty axes case
            R"(.*NormalizeL2LayerTest.*axes=\(\).*)",

            // Not allowed dynamic loop tests on GPU
            R"(.*smoke_StaticShapeLoop_dynamic_exit.*)",
            // Not expected behavior
            R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*layout=(SCALAR|OIHW).*)",
            R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*CanSetInBlobWithDifferentLayouts.*layout=NHWC.*)",
            R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*CanSetOutBlobWithDifferentLayouts.*layout=(CN|HW).*)",
            R"(.*Behavior.*(Multi|Auto).*InferRequestSetBlobByType.*Batched.*)",
            R"(.*(Multi|Auto).*Behavior.*InferRequestIOBBlobTest.*canProcessDeallocatedOutputBlobAfterGetAndSetBlob.*)",
            // TODO Issue 100145
            R"(.*Behavior.*InferRequestIOBBlobTest.*canReallocateExternalBlobViaGet.*)",
            R"(.*Behavior.*OVInferRequestIOTensorTest.*canInferAfterIOBlobReallocation.*)",
            R"(.*Behavior.*OVInferRequestDynamicTests.*InferUpperBoundNetworkAfterIOTensorsReshaping.*)",
            R"(.*(Auto|Multi).*Behavior.*IncorrectConfigTests.*CanNotLoadNetworkWithIncorrectConfig.*)",
            // TODO: until issue is xxx-59670 is resolved
            R"(.*Gather8LayerTest.*)",
            // Not implemented yet:
            R"(.*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNet.*)",
            R"(.*Behavior.*ExecutableNetworkBaseTest.*canExport.*)",
            R"(.*OVExecutableNetworkBaseTest.*CanSetConfigToExecNet.*)",
            R"(.*OVExecutableNetworkBaseTest.*CanSetConfigToExecNetAndCheckConfigAndCheck.*)",
            // TODO: Issue 67408
            R"(.*smoke_LSTMSequenceCommonClip.*LSTMSequenceTest.*CompareWithRefs.*)",
            // Expected behavior. GPU plugin doesn't support i64 for eltwise power operation.
            R"(.*EltwiseLayerTest.*OpType=Pow.*NetType=i64.*)",
            // TODO: Issue: 68712
            R"(.*.MatMul.*CompareWithRefs.*IS0=\(1.5\)_IS1=\(1.5\).*transpose_a=0.*transpose_b=1.*CONSTANT.*FP16.*UNSPECIFIED.*UNSPECIFIED.*ANY.*)",
            // TODO: Issue 69187
            R"(smoke_PrePostProcess.*cvt_color_nv12.*)",
            // TODO: Issue 71215
            R"(smoke_PrePostProcess.*cvt_color_i420.*)",
            // Unsupported
            R"(smoke_Behavior/InferRequestSetBlobByType.setInputBlobsByType/BlobType=Batched_Device=GPU_Config=().*)",
            // TODO: Issue 72624
            R"(smoke_PrePostProcess.*resize_dynamic.*)",
            // Issue: CVS-66778
            R"(.*smoke_Auto_BehaviorTests.*DynamicOutputToDynamicInput.*)",
            R"(.*smoke_Auto_BehaviorTests.*DynamicInputToDynamicOutput.*)",
            // Dynamic batch allocates output using upper bound
            R"(.*smoke_BehaviorTests.*InferUpperBoundNetworkWithGetTensor.*)",
            // need dynamic shapes
            R"(.*RangeLayerTest.*)",
            // need dynamic rank
            R"(.*smoke.*BehaviorTests.*InferFullyDynamicNetworkWith(S|G)etTensor.*)",
            R"(.*smoke.*BehaviorTests.*DynamicOutputToDynamicInput.*)",
            R"(.*smoke.*BehaviorTests.*DynamicInputToDynamicOutput.*)",
            // Issue: 76197
            R"(.*registerPluginsXMLUnicodePath.*)",
            // Not supported yet
            R"(.*CompileModelCacheTestBase.*ConvBias.*)",
            R"(.*CompileModelCacheTestBase.*KSOFunction.*)",
            R"(.*LoadNetworkCacheTestBase.*)",
            // Issue: 83014
            R"(.*smoke_RemoteBlob.*canInferOnUserQueue.*)",
            // Issue: CVS-76980
            R"(.*smoke_Auto_BehaviorTests.*InferDynamicNetwork/.*)",
            // Issue: CVS-86976
            R"(.*smoke_VirtualPlugin_BehaviorTests.*LoadedRemoteContext.*)",
            // Issue: CVS-88667 - Need to verify hetero interoperability
            R"(.*nightly_OVClassHeteroExecutableNetworlGetMetricTest.*SUPPORTED_(CONFIG_KEYS|METRICS).*)",
            // TODO: Issue: 89555
            R"(.*CoreThreadingTests.*smoke.*Network.*)",
            // Assign-3/ReadValue-3 does not have evaluate() methods; ref implementation does not save the value across the inferences.
            R"(smoke_MemoryTestV3.*)",
            // Unsupported 8d tensors
            R"(smoke_Basic/SqueezeUnsqueezeLayerTest.CompareWithRefs/OpType=Unsqueeze_IS=\(1.1.1.1\)_Axes=\((0.1.2|0.2.3|1.2.3|0.1.2.3|)\)_.*)",
            // Issue: 90539
            R"(smoke_AutoBatch_BehaviorTests/OVInferRequestIOTensorTest.InferStaticNetworkSetInputTensor/targetDevice=BATCH.*)",
            // TODO: range input with one element should NOT be regarded as dynamic batch model in Program::IsDynBatchModel().
            R"(.*smoke_select_CompareWithRefsNumpy_dynamic_range.*)",
            R"(.*CachingSupportCase.*LoadNetworkCacheTestBase.*CompareWithRefImpl.*)",
#if defined(_WIN32) || defined(_WIN64)
            R"(.*KernelCachingSupportCase.*CanCreateCacheDirAndDumpBinariesUnicodePath.*)",
#endif
            R"(.*CachingSupportCase.*GPU.*CompileModelCacheTestBase.*CompareWithRefImpl.*)",
            // Currently 1D convolution has an issue
            R"(.*smoke_GroupConvolution1D_ExplicitPadding_Disabled.*)",
            R"(.*smoke_GroupConvolutionLayerGPUTest_dynamic1DSymPad_Disabled.*)",
            R"(.*smoke_ConvolutionLayerGPUTest_dynamic1DSymPad.*)",

            // Looks like the test is targeting CPU plugin and doesn't respect that execution graph may vary from plugin to plugin
            R"(.*ExecGraphSerializationTest.*)",
    };
}
