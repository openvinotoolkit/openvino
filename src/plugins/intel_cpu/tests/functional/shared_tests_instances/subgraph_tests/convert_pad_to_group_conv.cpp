// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/convert_pad_to_group_conv.hpp"

#include <vector>

using namespace ov::test;

namespace {
const std::vector<std::vector<int64_t>> pads_1d{{0, 0, 0}, {0, 0, 1}, {0, 2, 0}, {3, 0, 0}};

const std::vector<float> values{0., 1.};

INSTANTIATE_TEST_SUITE_P(smoke_Pad_1D,
                         ConvertPadToConvTests,
                         ::testing::Combine(::testing::Values(ov::Shape{1, 8, 64}),
                                            ::testing::ValuesIn(pads_1d),
                                            ::testing::ValuesIn(pads_1d),
                                            ::testing::ValuesIn(values),
                                            ::testing::Values(ov::op::PadMode::CONSTANT),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConvertPadToConvTests::getTestCaseName);

const std::vector<std::vector<int64_t>> pads_2d{{0, 0, 0, 0},
                                                {0, 0, 1, 2},
                                                {0, 0, 2, 1},
                                                {0, 0, 10, 10},
                                                {0, 0, 0, 4},
                                                {0, 0, 4, 0}};

INSTANTIATE_TEST_SUITE_P(smoke_Pad_2D,
                         ConvertPadToConvTests,
                         ::testing::Combine(::testing::Values(ov::Shape{1, 8, 64, 16}),
                                            ::testing::ValuesIn(pads_2d),
                                            ::testing::ValuesIn(pads_2d),
                                            ::testing::ValuesIn(values),
                                            ::testing::Values(ov::op::PadMode::CONSTANT),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConvertPadToConvTests::getTestCaseName);
}  // namespace
