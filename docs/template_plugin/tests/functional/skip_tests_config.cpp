// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        ".*ExclusiveAsyncRequests.*",
        ".*reusableCPUStreamsExecutor.*",
        R"(.*SplitLayerTest.*numSplits\=30.*)",
        // CVS-44775: for all cases below
        ".*Hetero.*",
        ".*QueryNetwork.*",
        ".*SetAffinityWithKSO.*",
        ".*queryNetworkResultContainAllAndOnlyInputLayers.*",
        R"(.*IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS.*)",
        R"(.*IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS.*/2)",
        R"(.*IEClassExecutableNetworkGetMetricTest_NETWORK_NAME.*/2)",
        R"(.*IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS.*/2)",
        ".*LoadNetworkActualHeteroDeviceNoThrow.*",
        ".*LoadNetworkActualHeteroDevice2NoThrow.*",
        ".*IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS.*",
        // CVS-44774
        ".*PreprocessTest.*",
    };
}