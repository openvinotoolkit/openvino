// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
            // Issues - 34059
            ".*BehaviorTests\\.pluginDoesNotChangeOriginalNetwork.*",
            //TODO: Issue: 34349
            R"(.*(IEClassLoadNetwork).*(QueryNetworkMULTIWithHETERONoThrow_V10|QueryNetworkHETEROWithMULTINoThrow_V10).*)",
            //TODO: Issue: 34748
            R"(.*(ComparisonLayerTest).*)",
            // TODO: Issue: 39014
            R"(.*CoreThreadingTestsWithIterations.*smoke_LoadNetwork.*)",
            // TODO: Issue: 39612
            R"(.*Interpolate.*cubic.*tf_half_pixel_for_nn.*FP16.*)",
            // Expected behavior
            R"(.*EltwiseLayerTest.*eltwiseOpType=Pow.*netPRC=I64.*)",
            R"(.*EltwiseLayerTest.*IS=\(.*\..*\..*\..*\..*\).*eltwiseOpType=Pow.*secondaryInputType=CONSTANT.*)",
    };
}
