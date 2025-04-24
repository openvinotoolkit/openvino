// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/split_concat_memory.hpp"

#include <vector>

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::i32,
    ov::element::f16,
    ov::element::i16,
    ov::element::u8,
    ov::element::i8,
};

const std::vector<ov::Shape> shapes = {
    {1, 8, 3, 2},
    {3, 8, 3, 2},
    {3, 8, 3},
    {3, 8},
};

INSTANTIATE_TEST_SUITE_P(smoke_CPU,
                         SplitConcatMemory,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         SplitConcatMemory::getTestCaseName);
}  // namespace
