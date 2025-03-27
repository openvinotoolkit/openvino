// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

#include "openvino/core/core_visibility.hpp"

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> retVector{
        R"(.*smoke_(Multi|Auto|Hetero)_BehaviorTests.*OVPropertiesTests.*SetCorrectProperties.*)",
        R"(.*smoke_(Multi|Auto|Hetero)_BehaviorTests.*OVPropertiesTests.*canSetPropertyAndCheckGetProperty.*)",
        R"(.*OVInferRequestCheckTensorPrecision.*get(Input|Output|Inputs|Outputs)From.*FunctionWith(Single|Several).*type=(u4|u1|i4|boolean).*)",
        R"(.*OVGetMetricPropsTest.*OVGetMetricPropsTest.*GetMetricAndPrintNoThrow_AVAILABLE_DEVICES.*)",
        // CACHE_MODE property is not supported on NPU
        R"(.*OVCompiledModelBaseTest.*import_from_.*_blob.*targetDevice=(HETERO.NPU).*)",
        R"(.*OVCompiledModelBaseTest.*compile_from_.*_blob.*targetDevice=(HETERO.NPU).*)",
        R"(.*OVCompiledModelBaseTest.*compile_from_cached_weightless_blob.*targetDevice=(HETERO.NPU).*)",
        R"(.*OVCompiledModelBaseTest.*use_blob_hint_.*targetDevice=CPU.*)",
    };
    return retVector;
}
