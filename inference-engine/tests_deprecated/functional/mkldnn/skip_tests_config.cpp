// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO: Issue: 32074
        R"(.*(smoke_MobileNet/ModelTransformationsTest.LPT/mobilenet_v2_tf_depthwise_batch1_inPluginDisabled_inTestDisabled_asymmetric).*)"
    };
}