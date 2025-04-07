// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/split.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using ov::test::SplitLayerTest;

namespace {

INSTANTIATE_TEST_SUITE_P(
    smoke_NumSplitsCheck,
    SplitLayerTest,
    ::testing::Combine(::testing::Values(1, 2, 3, 5, 6, 10, 30),
                       ::testing::Values(0, 1, 2, 3),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(ov::test::static_shapes_to_test_representation({{30, 30, 30, 30}})),
                       ::testing::Values(std::vector<size_t>({})),
                       ::testing::Values(ov::test::utils::DEVICE_TEMPLATE)),
    SplitLayerTest::getTestCaseName);

}  // namespace
