// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "single_op_tests/experimental_detectron_generate_proposals_single_image.hpp"

namespace {
using ov::test::ExperimentalDetectronGenerateProposalsSingleImageLayerTest;

const std::vector<float> min_size = { 0.0f, 0.1f };
const std::vector<float> nms_threshold = { 0.7f };
const std::vector<int64_t> post_nms_count = { 6 };
const std::vector<int64_t> pre_nms_count = { 14, 1000 };

const std::vector<ov::test::InputShape> input_shape = {
    // im_info / anchors / deltas / scores
    ov::test::static_shapes_to_test_representation({{3}, {36, 4}, {12, 2, 6}, {3, 2, 6}}),
};

INSTANTIATE_TEST_SUITE_P(
    smoke_ExperimentalDetectronGenerateProposalsSingleImageLayerTest_f16,
    ExperimentalDetectronGenerateProposalsSingleImageLayerTest,
    ::testing::Combine(
        ::testing::Values(input_shape),
        ::testing::ValuesIn(min_size),
        ::testing::ValuesIn(nms_threshold),
        ::testing::ValuesIn(post_nms_count),
        ::testing::ValuesIn(pre_nms_count),
        ::testing::Values(ov::element::f16),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ExperimentalDetectronGenerateProposalsSingleImageLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_ExperimentalDetectronGenerateProposalsSingleImageLayerTest_f32,
        ExperimentalDetectronGenerateProposalsSingleImageLayerTest,
        ::testing::Combine(
                ::testing::Values(input_shape),
                ::testing::ValuesIn(min_size),
                ::testing::ValuesIn(nms_threshold),
                ::testing::ValuesIn(post_nms_count),
                ::testing::ValuesIn(pre_nms_count),
                ::testing::Values(ov::element::f32),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
        ExperimentalDetectronGenerateProposalsSingleImageLayerTest::getTestCaseName);

} // namespace
