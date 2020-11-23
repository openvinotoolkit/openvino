// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/max_pool.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<MaxPoolSpecificParams> max_pool_only_test_cases = {
        MaxPoolSpecificParams{ { 1, 3, 32, 32 }, { 1, 1 },
                               { 0, 0 }, { 0, 0 }, { 2, 2 },
                               ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::SAME_LOWER},
        MaxPoolSpecificParams{ { 1, 3, 32, 32 }, { 1, 1 },
                               { 0, 0 }, { 0, 0 }, { 2, 2 },
                               ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::SAME_UPPER},
        MaxPoolSpecificParams{ { 1, 3, 32, 32 }, { 1, 1 },
                               { 0, 0 }, { 0, 0 }, { 2, 2 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER},
        MaxPoolSpecificParams{ { 1, 3, 32, 32 }, { 1, 1 },
                               { 0, 0 }, { 0, 0 }, { 2, 2 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER},
        MaxPoolSpecificParams{ { 32, 32 , 2, 2}, { 2, 2 },
                               { 0, 0 }, { 0, 0 }, { 2, 2 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER},
        MaxPoolSpecificParams{ { 32, 32 , 4, 4}, { 1, 1 },
                               { 0, 0 }, { 0, 0 }, { 2, 2 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER},
        MaxPoolSpecificParams{ { 32, 32 , 2, 2}, { 4, 4 },
                               { 1, 1 }, { 1, 1 }, { 2, 2 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER},
        MaxPoolSpecificParams{ { 32, 32, 2, 2}, { 2, 2 },
                               { 2, 2 }, { 2, 2 }, { 4, 4 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER},
        MaxPoolSpecificParams{ { 32, 32, 4, 4 }, { 2, 2 },
                               { 2, 2 }, { 2, 2 }, { 2, 2 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER},
        MaxPoolSpecificParams{ { 32, 32, 1 , 1}, { 2, 2 },
                               { 1, 1 }, { 1, 1 }, { 2, 2 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER},
        MaxPoolSpecificParams{ { 32, 32, 2, 2}, { 2, 2 },
                               { 1, 1 }, { 2, 2 }, { 2, 2 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER},
        MaxPoolSpecificParams{ { 32, 32, 2, 2, 1}, { 2, 2, 2 },
                               { 1, 1, 1 }, { 2, 2, 2 }, { 2, 2, 2 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER},
        MaxPoolSpecificParams{ { 32, 32, 2, 2, 4}, { 2, 2, 2},
                               { 1, 1, 1 }, { 2, 2, 2}, { 2, 2, 2 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER},
        MaxPoolSpecificParams{ { 32, 32, 1, 1, 2}, { 1, 1, 1 },
                               { 1, 1, 1}, { 2, 2, 2}, { 2, 2, 2 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER},
        MaxPoolSpecificParams{ { 32, 32, 1, 1, 1}, { 2, 2, 2},
                               { 1, 1, 1 }, { 2, 2, 2 }, { 2, 2, 2 },
                               ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_CASE_P(
        smoke_MaxPool, MaxPoolLayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(max_pool_only_test_cases),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(std::map<std::string, std::string>())),
        MaxPoolLayerTest::getTestCaseName);

}  // namespace
