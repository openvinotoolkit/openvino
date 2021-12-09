// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "subgraph_tests/reshape_squeeze_reshape_relu.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    std::vector<ShapeAxesTuple> inputs = {
            {{1, 1, 3}, {0, 1}},
            {{1, 3, 1}, {0, 2}},
            {{1, 3, 1}, {0}},
            {{3, 1, 1}, {1, 2}},
            {{3, 1, 1}, {2}},
            {{4, 1, 3, 1, 1, 3}, {1, 3}},
            {{4, 3}, {1, 2}},
            {{4, 3}, {0}},
            {{1}, {0}},
    };



    std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
    };
    const std::vector<ngraph::helpers::SqueezeOpType> opTypes = {
            ngraph::helpers::SqueezeOpType::SQUEEZE,
            ngraph::helpers::SqueezeOpType::UNSQUEEZE
    };

    INSTANTIATE_TEST_SUITE_P(DISABLED_reshape_squeeze_reshape_relu, ReshapeSqueezeReshapeRelu,
                            ::testing::Combine(
                                    ::testing::ValuesIn(inputs),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(opTypes)),
                            ReshapeSqueezeReshapeRelu::getTestCaseName);
}  // namespace
