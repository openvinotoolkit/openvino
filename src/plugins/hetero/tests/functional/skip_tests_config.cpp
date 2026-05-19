// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

#include "openvino/core/core_visibility.hpp"

const std::vector<std::regex>& disabled_test_patterns() {
    const static std::vector<std::regex> patterns{
        std::regex(R"(.*smoke_(Multi|Auto|Hetero)_BehaviorTests.*OVPropertiesTests.*SetCorrectProperties.*)"),
        std::regex(
            R"(.*smoke_(Multi|Auto|Hetero)_BehaviorTests.*OVPropertiesTests.*canSetPropertyAndCheckGetProperty.*)"),
        std::regex(
            R"(.*OVInferRequestCheckTensorPrecision.*get(Input|Output|Inputs|Outputs)From.*FunctionWith(Single|Several).*type=(u4|u1|i4|boolean).*)"),
        std::regex(R"(.*OVGetMetricPropsTest.*OVGetMetricPropsTest.*GetMetricAndPrintNoThrow_AVAILABLE_DEVICES.*)"),
        // CACHE_MODE property is not supported on NPU
        std::regex(R"(.*OVCompiledModelBaseTest.*import_from_.*_blob.*targetDevice=(HETERO.NPU).*)"),
        std::regex(R"(.*OVCompiledModelBaseTest.*compile_from_.*_blob.*targetDevice=(HETERO.NPU).*)"),
        std::regex(R"(.*OVCompiledModelBaseTest.*compile_from_cached_weightless_blob.*targetDevice=(HETERO.NPU).*)"),
        std::regex(R"(.*OVCompiledModelBaseTest.*use_blob_hint_.*targetDevice=CPU.*)"),
        // model import is not supported
        std::regex(R"(.*OVCompiledModelBaseTest.import_from_.*)"),
    };

    return patterns;
}
