// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO: FIX BUG 31661
        // TODO: support InferRequest in GNAPlugin
        ".*InferRequestTests\\.canRun3AsyncRequestsConsistentlyFromThreadsWithoutWait.*",
        // TODO: FIX BUG 23741
        ".*InferRequestTests\\.canRun3SyncRequestsConsistentlyFromThreads.*",
        // TODO: FIX BUG 59041
        ".*Behavior.*CallbackThrowException.*",
        // TODO: FIX BUG 32210
        R"(.*ActivationLayerTest.CompareWithRefs/(Sigmoid|Tanh|Exp|Log).*)",
        R"(.*ActivationFQSubgraph.*activation=(Exp|Log).*)",
        // TODO: Issue 32542
        R"(.*(EltwiseLayerTest).*eltwiseOpType=(Sum|Sub).*opType=SCALAR.*)",
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
        R"(.*VariableStateTest.inferreq_smoke_VariableState_2infers*.*)",
        // TODO: Issue 24839
        R"(.*ConvolutionLayerTest.CompareWithRefs.*D=\(1.3\).*)",
        R"(.*ConvolutionLayerTest.CompareWithRefs.*D=\(3.1\).*)",
        R"(.*ConstantResultSubgraphTest.*IS=\(2\.3\.4\.5\).*)",
        R"(.*ConstantResultSubgraphTest.*inPrc=(U8|I8|I32|U64|I64|BOOL).*)",
        // TODO: Issue 51528
        R"(.*CachingSupport.*_(u8|i16)_.*)",
        // TODO: Issue 57363 (Param -> Result subgraphs)
        R"(.*smoke_MemoryTest.*LOW_LATENCY.*iteration_count=1_.*)",
        // TODO: Issue 57368 (accuracy)
        R"(.*smoke_MemoryTest.*LOW_LATENCY.*IS=\(1.10\).*)",
        R"(.*smoke_MemoryTest.*iteration_count=3.*IS=\(1.10\).*)",
        R"(.*smoke_MemoryTest.*iteration_count=4.*IS=\(1.10\).*)",
        R"(.*smoke_MemoryTest.*iteration_count=10.*IS=\(1.10\).*)",
        R"(.*smoke_MemoryTest.*LOW_LATENCY.*iteration_count=10.*IS=\(1.2\).*)",
    };
}
