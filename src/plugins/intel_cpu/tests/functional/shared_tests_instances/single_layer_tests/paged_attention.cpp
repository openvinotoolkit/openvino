// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/paged_attention.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/single_op/paged_attention.hpp"

namespace {
using ov::test::PagedAttentionLayerTest;
const std::vector<ov::element::Type> inputPrecisions = {
    ov::element::f32,
};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttention_basic_static,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ov::test::static_shapes_to_test_representation({
                                                {2, 8},         // InputShape query,
                                                {2, 8},         // InputShape key,
                                                {2, 8},         // InputShape value,
                                                {3, 2, 32, 4},  // InputShape key_cache,
                                                {3, 2, 32, 4},  // InputShape value_cache
                                            })),
                                            ::testing::Values(ov::test::PagedAttentionIntVectorsStruct{
                                                {0, 1},
                                                {0, 1, 2},
                                                {0, 1, 2},
                                                {0, 1, 2},
                                            }),
                                            ::testing::Values(ov::test::PagedAttentionMiscInpStruct{{1}, 1, {2, 4}, 1}),
                                            ::testing::Values(std::nullopt),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         PagedAttentionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_PagedAttention_static_rotation,
    PagedAttentionLayerTest,
    ::testing::Combine(::testing::Values(ov::test::static_shapes_to_test_representation({
                           {2, 8},         // InputShape query,
                           {2, 8},         // InputShape key,
                           {2, 8},         // InputShape value,
                           {1, 2, 32, 4},  // InputShape key_cache,
                           {1, 2, 32, 4},  // InputShape value_cache
                       })),
                       ::testing::Values(ov::test::PagedAttentionIntVectorsStruct{
                           {0, 1},
                           {0, 1, 2},
                           {0},
                           {0, 0, 0},
                       }),
                       ::testing::Values(ov::test::PagedAttentionMiscInpStruct{{1}, 0, {0, 0}, 10}),
                       ::testing::Values(ov::test::PagedAttentionRotationStruct{{0, 0, 0, 0}, {4, 32}, {4, 4}}),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    PagedAttentionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttention_scale,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ov::test::static_shapes_to_test_representation({
                                                {2, 8},         // InputShape query,
                                                {2, 8},         // InputShape key,
                                                {2, 8},         // InputShape value,
                                                {3, 2, 32, 4},  // InputShape key_cache,
                                                {3, 2, 32, 4},  // InputShape value_cache
                                            })),
                                            ::testing::Values(ov::test::PagedAttentionIntVectorsStruct{
                                                {0, 1},
                                                {0, 1, 2},
                                                {0, 1, 2},
                                                {0, 1, 2},
                                            }),
                                            ::testing::ValuesIn({
                                                // Negative scale NaN?
                                                // ov::test::PagedAttentionMiscInpStruct{{-0.5}, 1, {2, 4}, 1},
                                                ov::test::PagedAttentionMiscInpStruct{{1.5}, 1, {2, 4}, 1},
                                                ov::test::PagedAttentionMiscInpStruct{{5}, 1, {2, 4}, 1},
                                                ov::test::PagedAttentionMiscInpStruct{{}, 1, {2, 4}, 1},
                                            }),
                                            ::testing::Values(std::nullopt),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         PagedAttentionLayerTest::getTestCaseName);
}  // namespace
