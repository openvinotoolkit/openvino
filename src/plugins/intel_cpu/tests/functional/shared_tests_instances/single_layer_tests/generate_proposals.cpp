// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/generate_proposals.hpp"

namespace {
using ov::test::GenerateProposalsLayerTest;
using ov::test::InputShape;

const std::vector<float> min_size = { 1 };
const std::vector<float> nms_threshold = { 0.699999988079071f };
const std::vector<int64_t> post_nms_count = { 6 };
const std::vector<int64_t> pre_nms_count = { 1000 };

const std::vector<std::vector<InputShape>> dynamicInputShape = {
    // im_info / anchors / deltas / scores
    ov::test::static_shapes_to_test_representation({{2, 3}, {2, 6, 3, 4}, {2, 12, 2, 6}, {2, 3, 2, 6}}),
    {
        {{-1, -1}, {{2, 3}}},
        {{-1, -1, -1, -1}, {{2, 6, 3, 4}}},
        {{-1, -1, -1, -1}, {{2, 12, 2, 6}}},
        {{-1, -1, -1, -1}, {{2, 3, 2, 6}}}
    },
    {
        {{{1, 3}, {3, 6}}, {{2, 3}}},
        {{{2, 4}, {6, 12}, {3, 6}, {4, 8}}, {{2, 6, 3, 4}}},
        {{{1, 3}, {12, 24}, {2, 4}, {6, 12}}, {{2, 12, 2, 6}}},
        {{{1, 3}, {3, 6}, {2, 4}, {6, 12}}, {{2, 3, 2, 6}}}
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_BasicTest,
    GenerateProposalsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShape),
        ::testing::ValuesIn(min_size),
        ::testing::ValuesIn(nms_threshold),
        ::testing::ValuesIn(post_nms_count),
        ::testing::ValuesIn(pre_nms_count),
        ::testing::ValuesIn({true, false}),
        ::testing::Values(ov::element::Type_t::f32),
        ::testing::Values(ov::element::Type_t::i32),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    GenerateProposalsLayerTest::getTestCaseName);
} // namespace
