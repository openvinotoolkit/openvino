// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/split.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::SplitLayerTest;

namespace {

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
        ov::element::u8
};

INSTANTIATE_TEST_SUITE_P(smoke_NumSplitsCheck, SplitLayerTest,
                        ::testing::Combine(
                                ::testing::Values(1, 2, 3, 5),
                                ::testing::Values(0, 1, 2, 3),
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::static_shapes_to_test_representation({{30, 30, 30, 30}})),
                                ::testing::Values(std::vector<size_t>({})),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        SplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_splitWithUnusedOutputsTest, SplitLayerTest,
                        ::testing::Combine(
                                ::testing::Values(5),
                                ::testing::Values(0),
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::static_shapes_to_test_representation({{30, 30, 30, 30}})),
                                ::testing::Values(std::vector<size_t>({0, 3})),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        SplitLayerTest::getTestCaseName);
}  // namespace
