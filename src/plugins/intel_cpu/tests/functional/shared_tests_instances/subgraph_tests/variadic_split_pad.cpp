// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/variadic_split_pad.hpp"

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> input_type = {ov::element::f32};

const std::vector<ov::Shape> shapes = {
    {1, 8, 3, 2},
    {3, 8, 8, 8},
};

const std::vector<std::vector<size_t>> connectedIndexes = {
    {0},
    {0, 2},
    {0, 1, 3},
    {0, 1, 1, 0},
    {0, 0, 0, 1},
};

const std::vector<std::vector<size_t>> numSplits = {{2, 2, 2, 2}, {1, 2, 4, 1}, {3, 2, 2, 1}};

const std::vector<std::vector<int64_t>> padsBegin = {
    {0, 0, 0, 0},
    {0, 0, 1, 1},
};

const std::vector<std::vector<int64_t>> padsEnd = {
    {0, 0, 0, 0},
    {0, 0, 1, 1},
};

const std::vector<ov::op::PadMode> padMode = {ov::op::PadMode::CONSTANT,
                                              ov::op::PadMode::EDGE,
                                              ov::op::PadMode::REFLECT,
                                              ov::op::PadMode::SYMMETRIC};

INSTANTIATE_TEST_SUITE_P(smoke_CPU,
                         VariadicSplitPad,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(1),
                                            ::testing::ValuesIn(numSplits),
                                            ::testing::ValuesIn(connectedIndexes),
                                            ::testing::ValuesIn(padsBegin),
                                            ::testing::ValuesIn(padsEnd),
                                            ::testing::ValuesIn(padMode),
                                            ::testing::ValuesIn(input_type),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         VariadicSplitPad::getTestCaseName);
}  // namespace
