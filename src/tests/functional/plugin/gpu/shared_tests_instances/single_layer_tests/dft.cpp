// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include <single_layer_tests/dft.hpp>
#include <vector>

namespace {

const std::vector<ngraph::helpers::DFTOpType> opTypes = {ngraph::helpers::DFTOpType::FORWARD,
                                                         ngraph::helpers::DFTOpType::INVERSE};
const std::vector<InferenceEngine::Precision> inputPrecision = {InferenceEngine::Precision::FP32,
                                                                InferenceEngine::Precision::FP16};
const auto combine = [](const std::vector<InferenceEngine::SizeVector>& inputShapes,
                        const std::vector<std::vector<int64_t>>& axes,
                        const std::vector<std::vector<int64_t>>& signalSizes) {
    return testing::Combine(testing::ValuesIn(inputShapes),
                            testing::ValuesIn(inputPrecision),
                            testing::ValuesIn(axes),
                            testing::ValuesIn(signalSizes),
                            testing::ValuesIn(opTypes),
                            testing::Values(CommonTestUtils::DEVICE_GPU));
};

using namespace LayerTestsDefinitions;

INSTANTIATE_TEST_SUITE_P(smoke_DFT_2d,
                         DFTLayerTest,
                         combine({{10, 2}, {1, 2}},  // input shapes
                                 {{0}, {-1}},        // axes
                                 {{}, {5}}),         // signal sizes
                         DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DFT_3d,
                         DFTLayerTest,
                         combine({{10, 4, 2}, {1, 17, 2}},  // input shapes
                                 {{0, 1}, {-1, -2}},        // axes
                                 {{}, {5, 2}}),             // signal sizes
                         DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DFT_4d,
                         DFTLayerTest,
                         combine({{10, 4, 8, 2}, {1, 17, 12, 2}},  // input shapes
                                 {{0, 1, 2}, {-1, -2, -3}},        // axes
                                 {{}, {5, 2, 5}}),                 // signal sizes
                         DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DFT_5d,
                         DFTLayerTest,
                         combine({{10, 4, 8, 2, 2}, {1, 17, 12, 1, 2}},  // input shapes
                                 {{0, 1, 2, 3}, {-1, -2, -3, -4}},       // axes
                                 {{}, {5, 2, 5, 20}}),                   // signal sizes
                         DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DFT_6d,
                         DFTLayerTest,
                         combine({{10, 4, 8, 2, 5, 2}, {1, 17, 12, 1, 7, 2}},  // input shapes
                                 {{0, 1, 2, 3, 4}, {-1, -2, -3, -4, -5}},      // axes
                                 {{}, {5, 2, 5, 20, 10}}),                     // signal sizes
                         DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DFT_6d_zero,
                         DFTLayerTest,
                         combine({{10, 4, 8, 2, 5, 2}, {1, 17, 12, 1, 7, 2}},  // input shapes
                                 {{}},                                         // axes
                                 {{}}),                                        // signal sizes
                         DFTLayerTest::getTestCaseName);

}  // namespace
