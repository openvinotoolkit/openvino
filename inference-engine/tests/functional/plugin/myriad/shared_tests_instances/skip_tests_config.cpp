// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
            ".*ActivationLayerTest\\.CompareWithRefs/Tanh.*netPRC=FP32.*",
            ".*ActivationLayerTest\\.CompareWithRefs/Exp.*netPRC=FP32.*",
            ".*ActivationLayerTest\\.CompareWithRefs/Log.*netPRC=FP32.*",
            ".*ActivationLayerTest\\.CompareWithRefs/Sigmoid.*netPRC=FP32.*",
            ".*ActivationLayerTest\\.CompareWithRefs/Relu.*netPRC=FP32.*",
    };
}
