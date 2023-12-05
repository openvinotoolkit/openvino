// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> disabled_items = {
        // TODO: for CVS-68949
        // Not implemented yet:
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNet.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canExport.*)",
        R"(.*OVExecutableNetworkBaseTest.*CanSetConfigToExecNet.*)",
        R"(.*OVExecutableNetworkBaseTest.*CanSetConfigToExecNetAndCheckConfigAndCheck.*)",
        // Not supported by TEMPLATE plugin
        R"(.*OVExecutableNetworkBaseTest.*CheckExecGraphInfo.*)",
        // Issue: 90539
        R"(.*OVInferRequestIOTensorTest.InferStaticNetworkSetChangedInputTensorThrow.*)",
        R"(.*OVInferRequestIOTensorTest.canInferAfterIOBlobReallocation.*)",
        R"(.*VirtualPlugin.*BehaviorTests.*OVHoldersTest.*)",
        // BATCH plugin doesn't support this case
        R"(.*LoadNetworkCreateDefaultExecGraphResult.*)",
        // BATCH/TEMPLATE plugin doesn't support this case
        R"(.*OVInferRequestPerfCountersTest.*CheckOperationInProfilingInfo.*)",
    };

    return disabled_items;
}
