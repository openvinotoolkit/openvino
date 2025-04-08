// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/reshape.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::ReshapeLayerTest;

namespace {
const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16
};

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCheck, ReshapeLayerTest,
        ::testing::Combine(
                ::testing::Values(true),
                ::testing::ValuesIn(model_types),
                ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
                ::testing::Values(std::vector<int64_t>({10, 0, 100})),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCheckNegative, ReshapeLayerTest,
        ::testing::Combine(
                ::testing::Values(true),
                ::testing::ValuesIn(model_types),
                ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
                ::testing::Values(std::vector<int64_t>({10, -1, 100})),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ReshapeLayerTest::getTestCaseName);
}  // namespace
