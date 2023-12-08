// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO: FIX BUG 31661
        // TODO: support InferRequest in GNAPlugin
        ".*InferRequestMultithreadingTests\\.canRun3AsyncRequestsConsistentlyFromThreadsWithoutWait.*",
        // TODO: FIX BUG 23741
        ".*InferRequestMultithreadingTests\\.canRun3SyncRequestsConsistentlyFromThreads.*",
        // TODO: FIX BUG 59041
        ".*Behavior.*CallbackThrowException.*",
        // TODO: FIX BUG 32210
        R"(.*ActivationFQSubgraph.*activation=(Exp|Log).*)",
        // TODO: Issue 32542
        R"(.*(EltwiseLayerTest).*eltwiseOpType=(Sum|Sub).*opType=SCALAR.*)",
        // TODO: Issue 32541
        R"(.*(EltwiseLayerTest).*eltwiseOpType=Prod.*secondaryInputType=PARAMETER.*opType=SCALAR.*)",
        // TODO: Issue: 34348
        R"(.*IEClassGetAvailableDevices.*)",
        // TODO: Issue 39358
        R"(.*unaligned.*MultipleConcatTest.*)",
        R"(.*ActivationConcatsEltwise.*CS=35.*)",
        // TODO: Issue 38974
        R"(.*ConcatMultiInput.CompareWithRefConstOnly.*IS=\(1.8\).*)",
        R"(.*ConcatMultiInput.CompareWithRefConstOnly.*IS=\(1.16\).*)",
        R"(.*ConcatMultiInput.CompareWithRefConstOnly.*IS=\(1.32\).*)",
        // TODO: Issue: 46416
        R"(.*InferRequestVariableStateTest.inferreq_smoke_VariableState_2infers*.*)",
        // TODO: Issue 24839
        R"(.*ConvolutionLayerTestFixture.CompareWithRefs.*D=\(1.3\).*)",
        R"(.*ConvolutionLayerTestFixture.CompareWithRefs.*D=\(3.1\).*)",
        R"(.*ConstantResultSubgraphTest.*IS=\(2\.3\.4\.5\).*)",
        R"(.*ConstantResultSubgraphTest.*inPrc=(U8|I8|I32|U64|I64|BOOL).*)",
        R"(.*importExportedFunctionParameterResultOnly.*)",
        R"(.*importExportedFunctionConstantResultOnly.*)",
        R"(.*importExportedIENetworkConstantResultOnly.*)",
        R"(.*importExportedIENetworkParameterResultOnly.*)",

        // Issue 57368 (accuracy)
        R"(.*smoke_MemoryTest.*transformation=LOW_LATENCY.*)",

        // Not implemented yet
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(canSetConfigToExecNet|canSetConfigToExecNetWithIncorrectConfig).*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*(canSetConfigToCompiledModel|canSetConfigToCompiledModelWithIncorrectConfig).*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(CheckExecGraphInfoBeforeExecution|CheckExecGraphInfoAfterExecution|CheckExecGraphInfoSerialization).*)",
        R"(.*Behavior.*OVCompiledModelBaseTestOptional.*(CheckExecGraphInfoBeforeExecution|CheckExecGraphInfoAfterExecution).*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canExport.*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*canExportModel.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(CanCreateTwoExeNetworksAndCheckFunction).*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*(canCreateTwoCompiledModelAndCheckTheir).*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(checkGetExecGraphInfoIsNotNullptr).*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*(checkGetExecGraphInfoIsNotNullptr).*)",
        // Not implemented yet (dynamic cases)
        R"(.*Behavior.*OVInferenceChaining.*(StaticOutputToDynamicInput).*)",
        R"(.*Behavior.*OVInferenceChaining.*(DynamicOutputToDynamicInput).*)",
        R"(.*Behavior.*OVInferenceChaining.*(DynamicInputToDynamicOutput).*)",
        R"(.*Behavior.*OVInferRequestDynamicTests.*)",
        // Not expected behavior
        R"(.*Behavior.*ExecNetSetPrecision.*canSetInputPrecisionForNetwork.*FP16.*)",
        R"(.*OVCompiledModelBaseTest.*canSetConfigToCompiledModel.*)",
        R"(.*OVCompiledModelBaseTest.*canGetInputsInfoAndCheck.*)",
        R"(.*OVCompiledModelBaseTest.*getOutputsFromSplitFunctionWithSeveralOutputs.*)",
        R"(.*OVCompiledModelBaseTest.*canCompileModelFromMemory.*)",
        R"(.*OVCompiledModelBaseTest.*CanSetOutputPrecisionForNetwork.*)",
        R"(.*OVCompiledModelBaseTest.*CanSetInputPrecisionForNetwork.*)",
        R"(.*OVCompiledModelBaseTest.*CanCreateTwoCompiledModelsAndCheckRuntimeModel.*)",
        R"(.*(OVClass|IEClass)HeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK.*GetMetricNoThrow.*)",
        R"(.*LoadNetwork*.*LoadNetwork(HETEROWithDeviceIDNoThrow|WithBigDeviceID|WithInvalidDeviceID)*.*)",
        R"(.*QueryNetwork*.*QueryNetwork(HETEROWithDeviceIDNoThrow|WithBigDeviceID|WithInvalidDeviceID)*.*)",
        R"(.*QueryModel*.*QueryModel(HETEROWithDeviceIDNoThrow|WithBigDeviceID|WithInvalidDeviceID)*.*)",
        R"(.*LoadNetworkTest.*QueryNetwork(MULTIWithHETERO|HETEROWithMULTI)NoThrow_V10.*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*get(Inputs|Outputs)FromFunctionWithSeveral(Inputs|Outputs).*)",
        // TODO: temporary disabled. Need to be enabled when PR 9282 is merged
        R"(.*OVCompiledGraphImportExportTest.*readFromV10IR.*)",
        // Issue connected with OV2.0
        R"(.*EltwiseLayerTest.*NetType=f16.*)",
        // TODO: Issue: 69639
        R"(.*EltwiseLayerTest.*OpType=Prod.*)",
        R"(.*EltwiseLayerTest.*OpType=Sum.*PARAMETER.*VECTOR.*)",
        // TODO: Issue:27391
        // TODO: Issue:28036
        R"(.*ActivationLayerGNATest.*(Log|Exp).*netPRC=(FP16|FP32).*)",
        // TODO: Issue: 71068
        R"(.*OVInferRequestCancellationTests.*)",
        // TODO: Issue: 95609
        R"(.*CompileModelCacheTestBase.*(ConvPoolRelu|TIwithLSTMcell1).*batch2.*)",
        R"(.*CompileModelCacheTestBase.*(SplitConvConcat|KSOFunction).*)",
        R"(.*CompileModelCacheTestBase.*(SingleConv|NestedSplitConvConcat).*)",
        R"(.*CompileModelCacheTestBase.*(Bias|ReadConcatSplitAssign).*)",
        // does not work due to GNA 3.0 convolution and other primitives limitations, partially can be resolved by
        // switching GNA library to GNA3.5
        R"(.*CachingSupportCase.*LoadNet.*(Bias|Split|Concat|KSO|SingleConv).*)",
        R"(.*CachingSupportCase.*LoadNet.*(ConvPoolRelu|TIwithLSTMcell1)_f32_batch2.*)",
        R"(.*smoke_Multi_BehaviorTests.*)",
        // unsupported metrics
        R"(.*smoke_MultiHeteroOVGetMetricPropsTest.*OVGetMetricPropsTest.*(AVAILABLE_DEVICES|OPTIMIZATION_CAPABILITIES|RANGE_FOR_ASYNC_INFER_REQUESTS|RANGE_FOR_STREAMS).*)",
        // TODO: Issue: 111556
        R"(.*SplitConvTest.CompareWithRefImpl.*IS=\(1.(128|256)\).*IC=4.*OC=4.*configItem=GNA_DEVICE_MODE_GNA_SW_FP32)",
        // TODO: Issue: 114149
        R"(.*smoke_Decompose2DConv.*)",
        // TODO: Issue: 123306
        R"(smoke_convert_matmul_to_fc/ConvertMatmulToFcWithTransposesPass.CompareWithRefImpl/netPRC=FP(32|16)_targetDevice=GNA__configItem=GNA_COMPACT_MODE_NO_configItem=GNA_DEVICE_MODE_GNA_SW_(FP32|EXACT)_IS=\(8.*)",
    };
}
