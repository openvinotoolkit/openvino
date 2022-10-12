// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

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
        // TODO: Issue 32923
        R"(.*IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK.*)",
        // TODO: Issue 39358
        R"(.*unaligned.*MultipleConcatTest.*)",
        R"(.*ActivationConcatsEltwise.*CS=35.*)",
        // TODO: Issue 38974
        R"(.*ConcatMultiInput.CompareWithRefConstOnly.*IS=\(1.8\).*)",
        R"(.*ConcatMultiInput.CompareWithRefConstOnly.*IS=\(1.16\).*)",
        R"(.*ConcatMultiInput.CompareWithRefConstOnly.*IS=\(1.32\).*)",
        // TODO: Issue: 29577
        R"(.*CoreThreadingTests.smoke_QueryNetwork.*)",
        //TODO: Issue: 46416
        R"(.*InferRequestVariableStateTest.inferreq_smoke_VariableState_2infers*.*)",
        // TODO: Issue 24839
        R"(.*ConvolutionLayerTest.CompareWithRefs.*D=\(1.3\).*)",
        R"(.*ConvolutionLayerTest.CompareWithRefs.*D=\(3.1\).*)",
        R"(.*ConstantResultSubgraphTest.*IS=\(2\.3\.4\.5\).*)",
        R"(.*ConstantResultSubgraphTest.*inPrc=(U8|I8|I32|U64|I64|BOOL).*)",
        R"(.*importExportedFunctionParameterResultOnly.*)",
        R"(.*importExportedFunctionConstantResultOnly.*)",
        R"(.*importExportedIENetworkConstantResultOnly.*)",
        R"(.*importExportedIENetworkParameterResultOnly.*)",

        // TODO: Issue 57368 (accuracy)
        R"(.*smoke_MemoryTest.*LOW_LATENCY.*IS=\(1.10\).*)",
        R"(.*smoke_MemoryTest.*iteration_count=3.*IS=\(1.10\).*)",
        R"(.*smoke_MemoryTest.*iteration_count=4.*IS=\(1.10\).*)",
        R"(.*smoke_MemoryTest.*iteration_count=10.*IS=\(1.10\).*)",
        R"(.*smoke_MemoryTest.*LOW_LATENCY.*iteration_count=10.*IS=\(1.2\).*)",
        // Not implemented yet
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(canSetConfigToExecNet|canSetConfigToExecNetWithIncorrectConfig).*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(CheckExecGraphInfoBeforeExecution|CheckExecGraphInfoAfterExecution|CheckExecGraphInfoSerialization).*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canExport.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(CanCreateTwoExeNetworksAndCheckFunction).*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(checkGetExecGraphInfoIsNotNullptr).*)",
        // Not implemented yet (dynamic cases)
        R"(.*Behavior.*OVInferenceChaining.*(StaticOutputToDynamicInput).*)",
        R"(.*Behavior.*OVInferenceChaining.*(DynamicOutputToDynamicInput).*)",
        R"(.*Behavior.*OVInferenceChaining.*(DynamicInputToDynamicOutput).*)",
        R"(.*Behavior.*OVInferRequestDynamicTests.*)",
        // Not expected behavior
        R"(.*Behavior.*ExecNetSetPrecision.*canSetInputPrecisionForNetwork.*FP16.*)",
        R"(.*OVExecutableNetworkBaseTest.*CanSetConfigToExecNet.*)",
        R"(.*OVExecutableNetworkBaseTest.*CanGetInputsInfoAndCheck.*)",
        R"(.*OVExecutableNetworkBaseTest.*getOutputsFromSplitFunctionWithSeveralOutputs.*)",
        R"(.*OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK.*GetMetricNoThrow.*)",
        R"(.*Behavior.*OVExecutableNetworkBaseTest.*get(Inputs|Outputs)FromFunctionWithSeveral(Inputs|Outputs).*)",
        // TODO: temporary disabled. Need to be enabled when PR 9282 is merged
        R"(.*OVExecGraphImportExportTest.*readFromV10IR.*)",
        // TODO: Issue: 29577
        R"(.*QueryNetwork.*)",
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
        // TODO: Issue: 71070
        R"(.*OVInferenceChaining.*(StaticOutputToStaticInput).*)"
    };
}
