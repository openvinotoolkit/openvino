// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
            ".*BehaviorPluginTestInferRequest\\.canRun3AsyncRequestsConsistentlyFromThreadsWithoutWait.*",

            // TODO: FIX BUG 23740
            ".*BehaviorPluginTestInferRequest\\.CanCreateTwoExeNetworks.*",

            // TODO: FIX BUG 26702
            ".*BehaviorPluginTestInferRequest\\.FailedAsyncInferWithNegativeTimeForWait.*",

            // TODO: FIX BUG 23741
            ".*BehaviorPluginTestInferRequest\\.canRun3SyncRequestsConsistentlyFromThreads.*",

            // TODO: FIX BUG 23742
            ".*BehaviorPluginTestInferRequest\\.canWaitWithotStartAsync.*",

            // TODO: FIX BUG 23743
            ".*BehaviorPluginTestInferRequest\\.returnDeviceBusyOnSetBlobAfterAsyncInfer.*",
            ".*BehaviorPluginTestInferRequest\\.returnDeviceBusyOnGetBlobAfterAsyncInfer.*",
            ".*BehaviorPluginTestInferRequest\\.returnDeviceBusyOnGetPerformanceCountAfterAsyncInfer.*",
            ".*BehaviorPluginTestInferRequest\\.returnDeviceBusyOnStartInferAfterAsyncInfer.*",
            ".*BehaviorPluginTestInferRequest\\.returnDeviceBusyOnGetUserDataAfterAsyncInfer.*",
            ".*BehaviorPluginTestInferRequest\\.returnDeviceBusyOnSetUserDataAfterAsyncInfer.*",
    };
}