// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/concat.hpp"
#include "common_test_utils/test_constants.hpp"


namespace {
using ov::test::ConcatLayerTest;

std::vector<int> axes = {-3, -2, -1, 0, 1, 2, 3};
std::vector<std::vector<ov::Shape>> shapes_static = {
        {{10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}}
};


std::vector<ov::element::Type> netPrecisions = {ov::element::f32,
                                                ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, ConcatLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axes),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_static)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConcatLayerTest::getTestCaseName);
}  // namespace
