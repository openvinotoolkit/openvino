// Copyright (C) 2018-2025 Intel Corporation
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
}  // namespace
