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
            // Issue: 122177
            R"(.*LSTMSequenceCommon.*LSTMSequenceTest.Inference.*CONVERT_TO_TI.*)",
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
            // Issue: 124060
            R"(.*smoke_GridSample/GridSampleLayerTest.Inference/.*model_type=f16.*)",
            // Issue: 119648
            R"(.*smoke_LPT/InterpolateTransformation.*)",
#if defined(_WIN32)
            R"(.*KernelCachingSupportCase.*CanCreateCacheDirAndDumpBinariesUnicodePath.*)",
#endif
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
    };
}
