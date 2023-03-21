/// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/eye.hpp"

using namespace LayerTestsDefinitions;

namespace {

TEST_P(EyeLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}

const std::vector<ov::element::Type_t> netPrecisions =
    {ElementType::f32, ElementType::f16, ElementType::i32, ElementType::i8, ElementType::u8, ElementType::i64};

const std::vector<std::vector<int>> eyePars = {
    // rows, cols, diag_shift
    {3, 3, 0},
    {3, 4, 1},
    {4, 3, 1},
    {3, 4, 0},
    {4, 3, 0},
    {3, 4, -1},
    {4, 3, -1},
    {3, 4, 10},
    {4, 4, -2},
};

// dummy parameter to prevent empty set of test cases
const std::vector<std::vector<int>> emptyBatchShape = {{0}};
const std::vector<std::vector<int>> batchShapes1D = {{3}, {2}, {1}, {0}};
const std::vector<std::vector<int>> batchShapes2D = {{3, 2}, {2, 1}, {0, 0}};
const std::vector<std::vector<int>> batchShapes3D = {{3, 2, 1}, {1, 1, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_Eye2D_WithNonScalar_Test,
                         EyeLayerTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<ov::Shape>>{{{1}, {1}, {1}}}),
                                            ::testing::ValuesIn(emptyBatchShape),
                                            ::testing::ValuesIn(eyePars),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                         EyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_1DBatch_Test,
                         EyeLayerTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<ov::Shape>>{
                                                {{1}, {1}, {1}, {1}}}),
                                            ::testing::ValuesIn(batchShapes1D),
                                            ::testing::ValuesIn(eyePars),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                         EyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_2DBatch_Test,
                         EyeLayerTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<ov::Shape>>{
                                                {{1}, {1}, {1}, {2}}}),
                                            ::testing::ValuesIn(batchShapes2D),
                                            ::testing::ValuesIn(eyePars),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                         EyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_3DBatch_Test,
                         EyeLayerTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<ov::Shape>>{
                                                {{1}, {1}, {1}, {3}}}),
                                            ::testing::ValuesIn(batchShapes3D),
                                            ::testing::ValuesIn(eyePars),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                         EyeLayerTest::getTestCaseName);

}  // namespace
