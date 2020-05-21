// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/equal.hpp"

#include "common_test_utils/test_constants.hpp"

#include <vector>

using namespace LayerTestsDefinitions;

namespace {

std::vector<std::vector<InferenceEngine::SizeVector>> inShapes = {
        {{200}, {200}},
        {{1000}, {1}},
        {{1, 256, 512}, {1, 256, 512}},
        {{1}, {1, 256, 512}},
};

INSTANTIATE_TEST_CASE_P(equalS32, EqualLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(inShapes),
                ::testing::Values(InferenceEngine::Precision::I32),
                ::testing::Values(InferenceEngine::Precision::I32),
                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
        EqualLayerTest::getTestCaseName);

}  // namespace
