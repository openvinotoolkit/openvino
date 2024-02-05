// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/generate_proposals.hpp"

namespace {
using ov::test::GenerateProposalsLayerTest;
using ov::test::InputShape;

const std::vector<float> min_size = { 1.0f, 0.0f };
const std::vector<float> nms_threshold = { 0.7f };
const std::vector<int64_t> post_nms_count = { 6 };
const std::vector<int64_t> pre_nms_count = { 14, 1000 };
constexpr size_t num_batches = 2;
constexpr size_t height = 2;
constexpr size_t width = 6;
constexpr size_t number_of_anchors = 3;

const std::vector<std::vector<InputShape>> input_shape = {
        // im_info / anchors / boxesdeltas / scores
        ov::test::static_shapes_to_test_representation({{num_batches, 3},
                                               {height, width, number_of_anchors, 4},
                                               {num_batches, number_of_anchors * 4, height, width},
                                               {num_batches, number_of_anchors, height, width}
                                        }),
};

INSTANTIATE_TEST_SUITE_P(
        smoke_GenerateProposalsLayerTest_f16,
        GenerateProposalsLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(input_shape),
                ::testing::ValuesIn(min_size),
                ::testing::ValuesIn(nms_threshold),
                ::testing::ValuesIn(post_nms_count),
                ::testing::ValuesIn(pre_nms_count),
                ::testing::ValuesIn({true}),
                ::testing::ValuesIn({ov::element::f16}),
                ::testing::ValuesIn({ov::element::i32, ov::element::i64}),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
        GenerateProposalsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_GenerateProposalsLayerTest_f32,
        GenerateProposalsLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(input_shape),
                ::testing::ValuesIn(min_size),
                ::testing::ValuesIn(nms_threshold),
                ::testing::ValuesIn(post_nms_count),
                ::testing::ValuesIn(pre_nms_count),
                ::testing::ValuesIn({false}),
                ::testing::ValuesIn({ov::element::f32}),
                ::testing::ValuesIn({ov::element::i32, ov::element::i64}),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
        GenerateProposalsLayerTest::getTestCaseName);

} // namespace
