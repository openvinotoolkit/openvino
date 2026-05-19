// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

const std::vector<std::regex>& disabled_test_patterns() {
    const static std::vector<std::regex> patterns{
        // TODO: for CVS-68949
        // Not implemented yet:
        std::regex(R"(.*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNet.*)"),
        std::regex(R"(.*Behavior.*ExecutableNetworkBaseTest.*canExport.*)"),
        std::regex(R"(.*OVExecutableNetworkBaseTest.*CanSetConfigToExecNet.*)"),
        std::regex(R"(.*OVExecutableNetworkBaseTest.*CanSetConfigToExecNetAndCheckConfigAndCheck.*)"),
        // Not supported by TEMPLATE plugin
        std::regex(R"(.*OVExecutableNetworkBaseTest.*CheckExecGraphInfo.*)"),
        // Issue: 90539
        std::regex(R"(.*OVInferRequestIOTensorTest.InferStaticNetworkSetChangedInputTensorThrow.*)"),
        std::regex(R"(.*OVInferRequestIOTensorTest.canInferAfterIOBlobReallocation.*)"),
        std::regex(R"(.*VirtualPlugin.*BehaviorTests.*OVHoldersTest.*)"),
        // BATCH plugin doesn't support this case
        std::regex(R"(.*LoadNetworkCreateDefaultExecGraphResult.*)"),
        // BATCH/TEMPLATE plugin doesn't support this case
        std::regex(R"(.*OVInferRequestPerfCountersTest.*CheckOperationInProfilingInfo.*)"),
        // requires export_model be implemented
        std::regex(R"(.*Behavior.*OVCompiledModelBaseTest.*import_from_weightless_blob.*targetDevice=(BATCH).*)"),
        std::regex(R"(.*Behavior.*OVCompiledModelBaseTest.*compile_from.*_blob.*targetDevice=(BATCH).*)"),
        std::regex(R"(.*Behavior.*OVCompiledModelBaseTest.*use_blob_hint.*targetDevice=(BATCH).*)"),
        // model import is not supported
        std::regex(R"(.*OVCompiledModelBaseTest.import_from_.*)")
    };

    return patterns;
}
