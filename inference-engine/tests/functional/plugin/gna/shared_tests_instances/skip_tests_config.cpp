// Copyright (C) 2020 Intel Corporation
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
        R"(.*(Sigmoid|Tanh|Exp|Log).*)"
                // TODO: Issue 32541
        R"(.*(EltwiseLayerTest).*eltwiseOpType=Prod.*secondaryInputType=PARAMETER.*)",
        // TODO: Issue 32542
        R"(.*(EltwiseLayerTest).*eltwiseOpType=Su.*opType=SCALAR.*)",
        // TODO: Issue 32521
        R"(.*(EltwiseLayerTest).*eltwiseOpType=Sub.*netPRC=FP16.*")",
        R"(.*(EltwiseLayerTest).*secondaryInputType=CONSTANT.*netPRC=FP16.*)",
        R"(.*(EltwiseLayerTest).*eltwiseOpType=Prod.*secondaryInputType=CONSTANT_opType=SCALAR_netPRC=FP32.*)",
    };
}
