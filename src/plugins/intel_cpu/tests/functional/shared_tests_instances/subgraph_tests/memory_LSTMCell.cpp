// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/memory_LSTMCell.hpp"

#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {

std::vector<ov::test::utils::MemoryTransformation> transformation{
    ov::test::utils::MemoryTransformation::NONE,
    ov::test::utils::MemoryTransformation::LOW_LATENCY_V2,
};

std::vector<size_t> input_sizes = {80, 32, 64, 100, 25};

std::vector<size_t> hidden_sizes = {
    128,
    200,
    300,
    24,
    32,
};

ov::AnyMap additional_config = {};

INSTANTIATE_TEST_SUITE_P(smoke_MemoryLSTMCellTest,
                         MemoryLSTMCellTest,
                         ::testing::Combine(::testing::ValuesIn(transformation),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(input_sizes),
                                            ::testing::ValuesIn(hidden_sizes),
                                            ::testing::Values(additional_config)),
                         MemoryLSTMCellTest::getTestCaseName);
}  // namespace test
}  // namespace ov
