// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <vector>
#include <map>

#include "single_layer_tests/add.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<std::size_t>> inputShapes = {
        {std::vector<std::size_t>({1, 30}), std::vector<std::size_t>({1, 30})}
};

INSTANTIATE_TEST_CASE_P(CompareWithRefs, AddLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(inputShapes),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(std::map<std::string, std::string>({}))),
                AddLayerTest::getTestCaseName);

}  // namespace
