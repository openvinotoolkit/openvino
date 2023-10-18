// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include <single_op_tests/dft.hpp>
#include <vector>

namespace {
using ov::test::DFTLayerTest;

const std::vector<ov::test::utils::DFTOpType> opTypes = {
    ov::test::utils::DFTOpType::FORWARD,
    ov::test::utils::DFTOpType::INVERSE,
};

const std::vector<ov::element::Type> inputPrecisions = {
    ov::element::f32,
    ov::element::f16,
};

const auto combine = [](const std::vector<std::vector<ov::Shape>>& inputShapes,
                        const std::vector<std::vector<int64_t>>& axes,
                        const std::vector<std::vector<int64_t>>& signalSizes) {
    return testing::Combine(testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
                            testing::ValuesIn(inputPrecisions),
                            testing::ValuesIn(axes),
                            testing::ValuesIn(signalSizes),
                            testing::ValuesIn(opTypes),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(smoke_DFT_2d,
                         DFTLayerTest,
                         combine({{{10, 2}}},   // input shapes
                                 {{0}},       // axes
                                 {{}, {3}}),  // signal sizes
                         DFTLayerTest::getTestCaseName);

// INSTANTIATE_TEST_SUITE_P(smoke_DFT_3d,
//                          DFTLayerTest,
//                          combine({{{10, 4, 2}}},    // input shapes
//                                  {{0, 1}},        // axes
//                                  {{}, {3, 10}}),  // signal sizes
//                          DFTLayerTest::getTestCaseName);

// INSTANTIATE_TEST_SUITE_P(smoke_DFT_4d,
//                          DFTLayerTest,
//                          combine({{{10, 4, 8, 2}}},    // input shapes
//                                  {{0, 1, 2}},        // axes
//                                  {{}, {3, 10, 8}}),  // signal sizes
//                          DFTLayerTest::getTestCaseName);

// INSTANTIATE_TEST_SUITE_P(smoke_DFT_4d_negative_reversed_axes,
//                          DFTLayerTest,
//                          combine({{{10, 4, 8, 2}}},    // input shapes
//                                  {{-1, -2, -3}},     // axes
//                                  {{}, {8, 10, 3}}),  // signal sizes
//                          DFTLayerTest::getTestCaseName);

// INSTANTIATE_TEST_SUITE_P(smoke_DFT_4d_single_axis,
//                          DFTLayerTest,
//                          combine({{{10, 4, 8, 2}}},        // input shapes
//                                  {{0}, {1}, {2}},        // axes
//                                  {{}, {1}, {5}, {20}}),  // signal sizes
//                          DFTLayerTest::getTestCaseName);

// INSTANTIATE_TEST_SUITE_P(smoke_DFT_5d,
//                          DFTLayerTest,
//                          combine({{{10, 4, 8, 2, 2}}},    // input shapes
//                                  {{0, 1, 2, 3}},        // axes
//                                  {{}, {3, 10, 8, 6}}),  // signal sizes
//                          DFTLayerTest::getTestCaseName);

// INSTANTIATE_TEST_SUITE_P(smoke_DFT_6d,
//                          DFTLayerTest,
//                          combine({{{10, 4, 8, 2, 5, 2}}},    // input shapes
//                                  {{0, 1, 2, 3, 4}},        // axes
//                                  {{}, {3, 10, 8, 6, 2}}),  // signal sizes
//                          DFTLayerTest::getTestCaseName);

// INSTANTIATE_TEST_SUITE_P(smoke_DFT_6d_zero,
//                          DFTLayerTest,
//                          combine({{{10, 4, 8, 2, 5, 2}}},  // input shapes
//                                  {{}},                   // axes
//                                  {{}}),                  // signal sizes
//                          DFTLayerTest::getTestCaseName);

}  // namespace
