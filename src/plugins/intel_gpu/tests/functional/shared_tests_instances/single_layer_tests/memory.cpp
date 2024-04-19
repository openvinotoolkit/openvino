// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/memory.h"

namespace {
using ov::test::MemoryLayerTest;
using ov::test::MemoryV3LayerTest;

const std::vector<ov::Shape> inShapes = {
        {1},
        {3},
        {3, 3, 3},
        {2, 3, 4, 5},
};

const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::i32,
        ov::element::f32,
};

const std::vector<int64_t> iterationCount {1, 3, 10};

INSTANTIATE_TEST_SUITE_P(smoke_MemoryTest, MemoryLayerTest,
        ::testing::Combine(
                ::testing::Values(ov::test::utils::MemoryTransformation::NONE),
                ::testing::ValuesIn(iterationCount),
                ::testing::ValuesIn(inShapes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
        MemoryLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MemoryTestV3, MemoryV3LayerTest,
        ::testing::Combine(
                ::testing::Values(ov::test::utils::MemoryTransformation::NONE),
                ::testing::ValuesIn(iterationCount),
                ::testing::ValuesIn(inShapes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
        MemoryLayerTest::getTestCaseName);
} // namespace
