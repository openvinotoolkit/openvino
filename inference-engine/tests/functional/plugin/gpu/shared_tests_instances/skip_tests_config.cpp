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
            // TODO: Issue: 40958
            R"(.*(ConstantResultSubgraphTest).*)",

            // TODO: Issue: 41467 -- "unsupported element type f16 op Convert"
            R"(.*(ConvertLayerTest).*targetPRC=FP16.*)",
            // TODO: Issue: 41466 -- "Unsupported op 'ConvertLike'"
            R"(.*(ConvertLikeLayerTest).*)",
            // TODO: Issue: 41462
            R"(.*(SoftMaxLayerTest).*axis=0.*)",
            // TODO: Issue: 41461
            R"(.*TopKLayerTest.*k=10.*mode=min.*sort=index.*)",
            R"(.*TopKLayerTest.*k=5.*sort=(none|index).*)",
            // TODO: Issue: 42029
            R"(.*ExecGraphTests.CheckExecGraphInfoSerialization)",
    };
}
