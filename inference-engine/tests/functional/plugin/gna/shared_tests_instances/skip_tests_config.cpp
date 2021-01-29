// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        ".*TensorNamesTest\\.CheckAddOutput.*",
        // TODO: FIX BUG 31661
        // TODO: support InferRequest in GNAPlugin
        ".*InferRequestTests\\.canRun3AsyncRequestsConsistentlyFromThreadsWithoutWait.*",
        // TODO: FIX BUG 23740
        ".*InferRequestTests\\.CanCreateTwoExeNetworks.*",
        // TODO: FIX BUG 26702
        ".*InferRequestTests\\.FailedAsyncInferWithNegativeTimeForWait.*",
        // TODO: FIX BUG 23741
        ".*InferRequestTests\\.canRun3SyncRequestsConsistentlyFromThreads.*",
        // TODO: FIX BUG 23742
        ".*InferRequestTests\\.canWaitWithotStartAsync.*",
        // TODO: FIX BUG 23743
        ".*InferRequestTests\\.returnDeviceBusyOnSetBlobAfterAsyncInfer.*",
        ".*InferRequestTests\\.returnDeviceBusyOnGetBlobAfterAsyncInfer.*",
        ".*InferRequestTests\\.returnDeviceBusyOnGetPerformanceCountAfterAsyncInfer.*",
        ".*InferRequestTests\\.returnDeviceBusyOnStartInferAfterAsyncInfer.*",
        ".*InferRequestTests\\.returnDeviceBusyOnGetUserDataAfterAsyncInfer.*",
        ".*InferRequestTests\\.returnDeviceBusyOnSetUserDataAfterAsyncInfer.*",
        // TODO: FIX BUG 31661
        ".*InferRequestTests\\.canStartSeveralAsyncInsideCompletionCallbackNoSafeDtorWithoutWait.*",
        // TODO: FIX BUG 31661
        ".*Behavior.*CallbackThrowException.*",
        // TODO: FIX BUG 32210
        R"(.*ActivationLayerTest.CompareWithRefs/(Sigmoid|Tanh|Exp|Log).*)",
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
        // TODO: Issue: 40960
        R"(.*(ConstantResultSubgraphTest).*)",
        // TODO: Issue: 29577
        R"(.*CoreThreadingTests.smoke_QueryNetwork.*)",
        //TODO: Issue: 46416
        R"(.*VariableStateTest.inferreq_smoke_VariableState_2infers*.*)",
        // TODO: Issue 24839
        R"(.*ConvolutionLayerTest.CompareWithRefs.*D=\(1.3\).*)",
        R"(.*ConvolutionLayerTest.CompareWithRefs.*D=\(3.1\).*)"
    };
}
