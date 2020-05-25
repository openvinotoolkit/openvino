// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/multiply.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace MultiplyTestDefinitions;

namespace {
std::vector<std::vector<std::vector<size_t>>> inShapes = {
        {{2}},
        {{1, 1, 1, 3}},
        {{1, 2, 4}},
        {{1, 4, 4}},
        {{1, 4, 4, 1}},
        {{1, 1, 1, 1, 1, 1, 3}},
        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
};

std::vector<InferenceEngine::Precision> netPrecisions = { InferenceEngine::Precision::FP32,
                                                            InferenceEngine::Precision::FP16,
};

std::vector<SecondaryInputType> secondaryInputTypes = { SecondaryInputType::CONSTANT,
                                                        SecondaryInputType::PARAMETER,
};

std::vector<MultiplicationType> multiplicationTypes = { MultiplicationType::SCALAR,
                                                        MultiplicationType::VECTOR,
};

std::map<std::string, std::string> additional_config = {};

const auto multiply_params = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(secondaryInputTypes),
    ::testing::ValuesIn(multiplicationTypes),
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(CommonTestUtils::DEVICE_CPU),
    ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(multilpy, MultiplyLayerTest, multiply_params, MultiplyLayerTest::getTestCaseName);
}  // namespace
