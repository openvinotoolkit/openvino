// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"
#include "device_utils.hpp"

#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> disabled_items = {
        // TODO: for 22.2 (CVS-68949)
        R"(smoke_AutoBatching_CPU/AutoBatching_Test_DetectionOutput.*)",
        // Not implemented yet:
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNet.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canExport.*)",
        R"(.*OVExecutableNetworkBaseTest.*CanSetConfigToExecNet.*)",
        R"(.*OVExecutableNetworkBaseTest.*CanSetConfigToExecNetAndCheckConfigAndCheck.*)",
        // Issue: 90539
        R"(smoke_AutoBatch_BehaviorTests/OVInferRequestIOTensorTest.InferStaticNetworkSetInputTensor/targetDevice=BATCH.*)",
        R"(.*VirtualPlugin.*BehaviorTests.*OVHoldersTest.*)",
    };

    if (!ov::device_utils::is_device_supported("GPU")) {
        disabled_items.push_back(R"(.*GPU.*)");
    }

    if (!ov::device_utils::is_device_supported("TEMPLATE")) {
        disabled_items.push_back(R"(.*TEMPLATE.*)");
    }

    return disabled_items;
}
