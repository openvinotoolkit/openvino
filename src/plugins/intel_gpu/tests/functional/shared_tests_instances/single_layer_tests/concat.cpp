// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/concat.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ConcatLayerTest;

std::vector<int> axes = {-3, -2, -1, 0, 1, 2, 3};
std::vector<std::vector<ov::Shape>> inShapes = {
        {{10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}}
};
std::vector<ov::element::Type> netPrecisions = {ov::element::f32,
                                                ov::element::f16,
                                                ov::element::i64};


INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, ConcatLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axes),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConcatLayerTest::getTestCaseName);

std::vector<int> axes6D = {-6, -3, -1, 0, 1, 2, 3, 4, 5};
std::vector<std::vector<ov::Shape>> inShapes6D = {
        {{2, 4, 2, 3, 2, 3}, {2, 4, 2, 3, 2, 3}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat6D, ConcatLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axes6D),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes6D)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConcatLayerTest::getTestCaseName);

std::vector<int> axes7D = {-7, -3, -1, 0, 1, 2, 3, 4, 5, 6};
std::vector<std::vector<ov::Shape>> inShapes7D = {
        {{2, 4, 2, 3, 2, 3, 2}, {2, 4, 2, 3, 2, 3, 2}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat7D, ConcatLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axes7D),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes7D)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConcatLayerTest::getTestCaseName);

std::vector<int> axes8D = {-8, -3, -1, 0, 1, 2, 3, 4, 5, 6, 7};
std::vector<std::vector<ov::Shape>> inShapes8D = {
        {{2, 4, 2, 2, 2, 2, 2, 2}, {2, 4, 2, 2, 2, 2, 2, 2}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat8D, ConcatLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axes8D),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes8D)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConcatLayerTest::getTestCaseName);
}  // namespace
