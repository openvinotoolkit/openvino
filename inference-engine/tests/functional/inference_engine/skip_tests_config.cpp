// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO: FIX BUG 33375
        // Disabled due to rare sporadic failures.
        ".*TransformationTests\\.ConstFoldingPriorBoxClustered.*",

        // LPT to nGraph migration: temporary disabling unexpected not reproduced fails on CI:
        // https://openvino-ci.intel.com/job/private-ci/job/ie/job/build-linux-ubuntu18_i386/478/
        "*ConvertNMS4ToNMSIEDynamic1*",
        "*ConvertNMS4ToNMSIEDynamic2*",
        "*MishFusing*"
        "*ConcatTransformation*",
        "*ConcatWithIntermediateTransformation*",
        "*ConcatWithNeighborsTransformation*"

        // TODO: task 32568, enable after supporting constants outputs in plugins
        ".*TransformationTests\\.ConstFoldingPriorBox.*",
    };
}
