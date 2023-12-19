// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_layer_tests/group_normalization.hpp"

using namespace ov::test::subgraph;

namespace {

const std::vector<ov::test::ElementType> netPrecisions = {
    ov::element::f32,
    ov::element::bf16,
    ov::element::i8
};

const std::vector<ov::test::InputShape> inputShapes = {
    // static shape
    {{1, 4, 1, 8}, {{1, 4, 1, 8}}},
    {{3, 8, 2, 32}, {{3, 8, 2, 32}}},
    {{3, 8, 16, 8, 4}, {{3, 8, 16, 8, 4}}},
    {{3, 8, 16, 8, 64}, {{3, 8, 16, 8, 64}}},
    {{3, 8, 16, 100, 4}, {{3, 8, 16, 100, 4}}},
    {{3, 16, 16, 8, 4}, {{3, 16, 16, 8, 4}}},
    {{1, 8, 8}, {{1, 8, 8}}},
    {{1, 8, 1, 8, 2}, {{1, 8, 1, 8, 2}}},
    {{1, 8, 1, 8, 2, 2}, {{1, 8, 1, 8, 2, 2}}},
    {{1, 8, 1, 8, 2, 2, 2}, {{1, 8, 1, 8, 2, 2, 2}}},
    // dynmaic shape
    {{-1, -1, -1, -1}, {{1, 16, 8, 8}, {2, 8, 4, 4}, {1, 16, 8, 8}}},
    {{{1, 4}, 16, -1, -1}, {{1, 16, 6, 6}, {4, 16, 10, 10}, {1, 16, 6, 6}}}
};

const std::vector<int64_t> numGroups = {
    2, 4,
};

const std::vector<double> epsilon = {
    0.0001
};

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupNormalization,
    GroupNormalizationTest,
    testing::Combine(testing::ValuesIn(netPrecisions),
                     ::testing::Values(ov::element::undefined),
                     ::testing::Values(ov::element::undefined),
                     testing::ValuesIn(inputShapes),
                     testing::ValuesIn(numGroups),
                     testing::ValuesIn(epsilon),
                     testing::Values(ov::test::utils::DEVICE_CPU),
                     testing::Values(ov::AnyMap())),
                     GroupNormalizationTest::getTestCaseName);

} // anonymous namespace