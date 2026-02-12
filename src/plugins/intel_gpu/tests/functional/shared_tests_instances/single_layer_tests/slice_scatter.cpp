// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/slice_scatter.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using ov::test::SliceScatterLayerTest;
using ov::test::SliceScatterSpecificParams;

namespace {

const std::vector<ov::element::Type> model_types = {ov::element::f32, ov::element::f16, ov::element::i8};

const std::vector<ov::element::Type> model_types_extra = {ov::element::i64, ov::element::i32, ov::element::u8};

std::vector<SliceScatterSpecificParams> static_params = {
    SliceScatterSpecificParams{{{{}, {{16}}}, {{}, {{8}}}}, {4}, {12}, {1}, {0}},
    SliceScatterSpecificParams{{{{}, {{16}}}, {{}, {{4}}}}, {0}, {8}, {2}, {0}},
    SliceScatterSpecificParams{{{{}, {{20, 10, 5}}}, {{}, {{20, 10, 5}}}}, {0, 0}, {10, 20}, {1, 1}, {1, 0}},
    SliceScatterSpecificParams{{{{}, {{1, 2, 12, 100}}}, {{}, {{1, 1, 5, 10}}}},
                               {0, 1, 0, 1},
                               {1, 2, 5, 100},
                               {1, 1, 1, 10},
                               {}},
    SliceScatterSpecificParams{{{{}, {{1, 12, 100}}}, {{}, {{1, 2, 1}}}}, {0, 9, 0}, {1, 11, 1}, {1, 1, 1}, {0, 1, -1}},
    SliceScatterSpecificParams{{{{}, {{2, 12, 100}}}, {{}, {{1, 4, 99}}}}, {1, 12, 100}, {0, 7, 0}, {-1, -1, -1}, {}},
    SliceScatterSpecificParams{{{{}, {{2, 12, 100}}}, {{}, {{1, 3, 99}}}}, {1, 4, 99}, {0, 9, 0}, {-1, 2, -1}, {}},
    SliceScatterSpecificParams{{{{}, {{2, 2, 2, 2}}}, {{}, {{2, 2, 2, 2}}}},
                               {0, 0, 0, 0},
                               {2, 2, 2, 2},
                               {1, 1, 1, 1},
                               {}},
    SliceScatterSpecificParams{{{{}, {{2, 2, 2, 2}}}, {{}, {{1, 1, 1, 1}}}},
                               {1, 1, 1, 1},
                               {2, 2, 2, 2},
                               {1, 1, 1, 1},
                               {}},
    SliceScatterSpecificParams{{{{}, {{2, 2, 4, 3}}}, {{}, {{2, 2, 2, 3}}}},
                               {0, 0, 0, 0},
                               {2, 2, 4, 3},
                               {1, 1, 2, 1},
                               {-4, 1, -2, 3}},
    SliceScatterSpecificParams{{{{}, {{1, 5, 32, 32}}}, {{}, {{1, 2, 23, 23}}}},
                               {0, 2, 5, 4},
                               {1, 4, 28, 27},
                               {1, 1, 1, 1},
                               {0, 1, 2, 3}},
    SliceScatterSpecificParams{{{{}, {{2, 5, 32, 32}}}, {{}, {{1, 2, 15, 10}}}},
                               {0, 0, 0, 20},
                               {1, 2, 30, 30},
                               {1, 1, 2, 1},
                               {0, 1, 2, 3}},
    SliceScatterSpecificParams{{{{}, {{10, 12}}}, {{}, {{10, 9}}}}, {-1, 1}, {-9999, 10}, {-1, 1}, {}},
    SliceScatterSpecificParams{{{{}, {{2, 4, 5, 5, 68}}}, {{}, {{2, 3, 5, 5, 5}}}},
                               {0, 1, 0, 0, 0},
                               {std::numeric_limits<std::int64_t>::max(),
                                std::numeric_limits<std::int64_t>::max(),
                                std::numeric_limits<std::int64_t>::max(),
                                std::numeric_limits<std::int64_t>::max(),
                                std::numeric_limits<std::int64_t>::max()},
                               {1, 1, 1, 1, 16},
                               {}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GPU,
                         SliceScatterLayerTest,
                         ::testing::Combine(::testing::ValuesIn(static_params),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         SliceScatterLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPU_PrecisionTransformation,
                         SliceScatterLayerTest,
                         ::testing::Combine(::testing::Values(static_params[0]),
                                            ::testing::ValuesIn(model_types_extra),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         SliceScatterLayerTest::getTestCaseName);

std::vector<SliceScatterSpecificParams> dynamic_params = {
    SliceScatterSpecificParams{{{{-1}, {{8}, {16}}}, {{-1}, {{4}, {8}}}}, {4}, {12}, {1}, {0}},
    SliceScatterSpecificParams{{{{ov::Dimension(2, 20)}, {{5}, {15}}}, {{ov::Dimension(2, 20)}, {{3}, {4}}}},
                               {0},
                               {8},
                               {2},
                               {0}},
    SliceScatterSpecificParams{{{{-1, -1, -1, -1}, {{1, 2, 12, 100}}}, {{-1, -1, -1, -1}, {{1, 1, 5, 10}}}},
                               {0, 1, 0, 1},
                               {1, 2, 5, 100},
                               {1, 1, 1, 10},
                               {}},
    SliceScatterSpecificParams{{{{ov::Dimension(1, 5), ov::Dimension(1, 5), ov::Dimension(1, 5), ov::Dimension(1, 5)},
                                 {{2, 2, 2, 2}, {2, 2, 4, 3}, {2, 2, 4, 2}, {1, 2, 4, 2}}},
                                {{ov::Dimension(1, 5), ov::Dimension(1, 5), ov::Dimension(1, 5), ov::Dimension(1, 5)},
                                 {{2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {1, 2, 2, 2}}}},
                               {0, 0, 0, 0},
                               {2, 2, 2, 2},
                               {1, 1, 1, 1},
                               {}},
    SliceScatterSpecificParams{{{{ov::Dimension(1, 5), ov::Dimension(1, 7), ov::Dimension(1, 35), ov::Dimension(1, 35)},
                                 {{1, 5, 16, 32}, {2, 5, 32, 20}, {2, 5, 32, 32}}},
                                {{ov::Dimension(1, 5), ov::Dimension(1, 7), ov::Dimension(1, 35), ov::Dimension(1, 35)},
                                 {{1, 2, 11, 23}, {1, 2, 23, 16}, {1, 2, 23, 23}}}},
                               {0, 2, 5, 4},
                               {1, 4, 28, 27},
                               {1, 1, 1, 1},
                               {0, 1, 2, 3}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GPU_dynamic,
                         SliceScatterLayerTest,
                         ::testing::Combine(::testing::ValuesIn(dynamic_params),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         SliceScatterLayerTest::getTestCaseName);

}  // namespace
