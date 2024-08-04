// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_op_tests/memory.h"

namespace {
using ov::test::MemoryLayerTest;

std::vector<ov::test::utils::MemoryTransformation> transformation {
        ov::test::utils::MemoryTransformation::NONE,
        ov::test::utils::MemoryTransformation::LOW_LATENCY_V2,
        ov::test::utils::MemoryTransformation::LOW_LATENCY_V2_ORIGINAL_INIT,
};

const std::vector<ov::Shape> inShapes = {
        {3},
        {100, 100},
};

const std::vector<ov::element::Type> input_types = {
        ov::element::f32,
};

const std::vector<int64_t> iterationCount{1, 3, 10};

INSTANTIATE_TEST_SUITE_P(smoke_MemoryTest, MemoryLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(transformation),
                ::testing::ValuesIn(iterationCount),
                ::testing::ValuesIn(inShapes),
                ::testing::ValuesIn(input_types),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        MemoryLayerTest::getTestCaseName);

}  // namespace

