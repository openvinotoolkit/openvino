// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/experimental_detectron_generate_proposals_single_image.hpp"

namespace {
using ov::test::ExperimentalDetectronGenerateProposalsSingleImageLayerTest;

const std::vector<float> min_size = { 0 };
const std::vector<float> nms_threshold = { 0.699999988079071 };
const std::vector<int64_t> post_nms_count = { 6 };
const std::vector<int64_t> pre_nms_count = { 1000 };

const std::vector<std::vector<ov::test::InputShape>> dynamicInputShape = {
    // im_info / anchors / deltas / scores
    ov::test::static_shapes_to_test_representation({{3}, {36, 4}, {12, 2, 6}, {3, 2, 6}}),
    {
        {{-1}, {{3}}},
        {{-1, -1}, {{36, 4}}},
        {{-1, -1, -1}, {{12, 2, 6}}},
        {{-1, -1, -1}, {{3, 2, 6}}}
    },
    {
        {{{3, 6}}, {{3}}},
        {{{36, 72}, {4, 8}}, {{36, 4}}},
        {{{12, 24}, {2, 4}, {6, 12}}, {{12, 2, 6}}},
        {{{3, 6}, {2, 4}, {6, 12}}, {{3, 2, 6}}}
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_ExperimentalDetectronGenerateProposalsSingleImageLayerTest,
    ExperimentalDetectronGenerateProposalsSingleImageLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShape),
        ::testing::ValuesIn(min_size),
        ::testing::ValuesIn(nms_threshold),
        ::testing::ValuesIn(post_nms_count),
        ::testing::ValuesIn(pre_nms_count),
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    ExperimentalDetectronGenerateProposalsSingleImageLayerTest::getTestCaseName);
} // namespace
