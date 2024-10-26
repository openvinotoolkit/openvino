// Copyright (C) 2018-2024 Intel Corporation
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
        R"(.*OVGetMetricPropsTest.*OVGetMetricPropsTest.*GetMetricAndPrintNoThrow_AVAILABLE_DEVICES.*)"};
    return retVector;
}
