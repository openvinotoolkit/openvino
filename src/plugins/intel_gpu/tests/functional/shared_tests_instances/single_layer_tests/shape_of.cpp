// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/shape_of.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> model_precisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32
};

const std::vector<InferenceEngine::SizeVector> input_shapes = {
    std::vector<size_t>({1, 2, 3, 4, 5}),
    std::vector<size_t>({1, 2, 3, 4}),
    std::vector<size_t>({1, 2})
};

INSTANTIATE_TEST_SUITE_P(smoke_Check, ShapeOfLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(model_precisions),
                                ::testing::Values(InferenceEngine::Precision::I64),
                                ::testing::ValuesIn(input_shapes),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ShapeOfLayerTest::getTestCaseName);
}  // namespace
