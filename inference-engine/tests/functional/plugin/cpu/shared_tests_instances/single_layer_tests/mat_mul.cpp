// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mat_mul.hpp"

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::MatMulParams;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<std::vector<size_t>> shapesA = {
        {1, 4, 5, 6}
};

const std::vector<std::vector<size_t>> shapesB = {
        {1, 4, 6, 4}
};

std::vector<InputLayerType> secondaryInputTypes = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER,
};

INSTANTIATE_TEST_CASE_P(MatMul, MatMulTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(shapesA),
                ::testing::ValuesIn(shapesB),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        MatMulTest::getTestCaseName);

} // namespace

