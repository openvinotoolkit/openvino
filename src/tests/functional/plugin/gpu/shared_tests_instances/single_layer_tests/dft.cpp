// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include <single_layer_tests/dft.hpp>
#include <vector>

namespace {

const std::vector<ngraph::helpers::DFTOpType> opTypes = {
    ngraph::helpers::DFTOpType::FORWARD,
    ngraph::helpers::DFTOpType::INVERSE,
};

const std::vector<ngraph::helpers::DFTOpMode> opModes = {
    ngraph::helpers::DFTOpMode::COMPLEX,
    ngraph::helpers::DFTOpMode::REAL,
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const auto combine = [](const std::vector<InferenceEngine::SizeVector>& inputShapes,
                        const std::vector<std::vector<int64_t>>& axes,
                        const std::vector<std::vector<int64_t>>& signalSizes) {
    return testing::Combine(testing::ValuesIn(inputShapes),
                            testing::ValuesIn(inputPrecisions),
                            testing::ValuesIn(axes),
                            testing::ValuesIn(signalSizes),
                            testing::ValuesIn(opTypes),
                            testing::ValuesIn(opModes),
                            testing::Values(CommonTestUtils::DEVICE_GPU));
};

using namespace LayerTestsDefinitions;

// RDFT can support 1d
INSTANTIATE_TEST_SUITE_P(smoke_DFT_1d_real,
                         DFT9LayerTest,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{10}),
                                          testing::ValuesIn(inputPrecisions),
                                          testing::Values(std::vector<int64_t>{0}),
                                          testing::Values(std::vector<int64_t>{}),
                                          testing::Values(ngraph::helpers::DFTOpType::FORWARD),
                                          testing::Values(ngraph::helpers::DFTOpMode::REAL),
                                          testing::Values(CommonTestUtils::DEVICE_GPU)),
                         DFT9LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_2d,
                         DFT9LayerTest,
                         combine({{10, 2}},   // input shapes
                                 {{0}},       // axes
                                 {{}, {5}}),  // signal sizes
                         DFT9LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_3d,
                         DFT9LayerTest,
                         combine({{10, 4, 2}},   // input shapes
                                 {{0, 1}},       // axes
                                 {{}, {5, 2}}),  // signal sizes
                         DFT9LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_4d,
                         DFT9LayerTest,
                         combine({{10, 4, 8, 2}},   // input shapes
                                 {{0, 1, 2}},       // axes
                                 {{}, {5, 2, 5}}),  // signal sizes
                         DFT9LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_4d_negative_axes,
                         DFT9LayerTest,
                         combine({{10, 4, 8, 2}},   // input shapes
                                 {{-1, -2, -3}},    // axes
                                 {{}, {5, 2, 5}}),  // signal sizes
                         DFT9LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_4d_single_axis,
                         DFT9LayerTest,
                         combine({{10, 4, 8, 2}},        // input shapes
                                 {{0}, {1}, {2}},        // axes
                                 {{}, {1}, {5}, {20}}),  // signal sizes
                         DFT9LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_5d,
                         DFT9LayerTest,
                         combine({{10, 4, 8, 2, 2}},    // input shapes
                                 {{0, 1, 2, 3}},        // axes
                                 {{}, {5, 2, 5, 20}}),  // signal sizes
                         DFT9LayerTest::getTestCaseName);

// RDFT can support last axis
INSTANTIATE_TEST_SUITE_P(smoke_DFT_5d_real_last_axis,
                         DFT9LayerTest,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{10, 4, 8, 2, 5}),
                                          testing::ValuesIn(inputPrecisions),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{{0, 1, 2, 3, 4}}}),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{}, {5, 2, 5, 20, 10}}),
                                          testing::Values(ngraph::helpers::DFTOpType::FORWARD),
                                          testing::Values(ngraph::helpers::DFTOpMode::REAL),
                                          testing::Values(CommonTestUtils::DEVICE_GPU)),
                         DFT9LayerTest::getTestCaseName);

// DFT, IDFT and IRDFT can support 6d
INSTANTIATE_TEST_SUITE_P(smoke_DFT_6d_complex,
                         DFT9LayerTest,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{10, 4, 8, 2, 5, 2}),
                                          testing::ValuesIn(inputPrecisions),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{{0, 1, 2, 3, 4}}}),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{}, {5, 2, 5, 20, 10}}),
                                          testing::Values(ngraph::helpers::DFTOpType::FORWARD,
                                                          ngraph::helpers::DFTOpType::INVERSE),
                                          testing::Values(ngraph::helpers::DFTOpMode::COMPLEX),
                                          testing::Values(CommonTestUtils::DEVICE_GPU)),
                         DFT9LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_6d_real,
                         DFT9LayerTest,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{10, 4, 8, 2, 5, 2}),
                                          testing::ValuesIn(inputPrecisions),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{{0, 1, 2, 3, 4}}}),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{}, {5, 2, 5, 20, 10}}),
                                          testing::Values(ngraph::helpers::DFTOpType::INVERSE),
                                          testing::Values(ngraph::helpers::DFTOpMode::REAL),
                                          testing::Values(CommonTestUtils::DEVICE_GPU)),
                         DFT9LayerTest::getTestCaseName);

// DFT and IDFT can support empty axes
INSTANTIATE_TEST_SUITE_P(smoke_DFT_6d_complex_empty_axes,
                         DFT9LayerTest,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{10, 4, 8, 2, 5, 2}),
                                          testing::ValuesIn(inputPrecisions),
                                          testing::Values(std::vector<int64_t>{}),
                                          testing::Values(std::vector<int64_t>{}),
                                          testing::Values(ngraph::helpers::DFTOpType::FORWARD,
                                                          ngraph::helpers::DFTOpType::INVERSE),
                                          testing::Values(ngraph::helpers::DFTOpMode::COMPLEX),
                                          testing::Values(CommonTestUtils::DEVICE_GPU)),
                         DFT9LayerTest::getTestCaseName);

}  // namespace
