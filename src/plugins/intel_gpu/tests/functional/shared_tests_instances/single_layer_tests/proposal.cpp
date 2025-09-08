// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/proposal.hpp"

namespace {
using ov::test::ProposalLayerTest;
using ov::test::proposalSpecificParams;
using ov::test::proposalLayerTestParamsSet;

/* ============= Proposal ============= */
const std::vector<size_t> base_size_ = {16};
const std::vector<size_t> pre_nms_topn_ = {100};
const std::vector<size_t> post_nms_topn_ = {100};
const std::vector<float> nms_thresh_ = {0.7f};
const std::vector<size_t> min_size_ = {1};
const std::vector<std::vector<float>> ratio_ = {{1.0f, 2.0f}};
const std::vector<std::vector<float>> scale_ = {{1.2f, 1.5f}};
const std::vector<bool> clip_before_nms_ = {false};
const std::vector<bool> clip_after_nms_ = {false};

// empty string corresponds to Caffe framework
const std::vector<std::string> framework_ = {""};

const auto proposalParams = ::testing::Combine(
        ::testing::ValuesIn(base_size_),
        ::testing::ValuesIn(pre_nms_topn_),
        ::testing::ValuesIn(post_nms_topn_),
        ::testing::ValuesIn(nms_thresh_),
        ::testing::ValuesIn(min_size_),
        ::testing::ValuesIn(ratio_),
        ::testing::ValuesIn(scale_),
        ::testing::ValuesIn(clip_before_nms_),
        ::testing::ValuesIn(clip_after_nms_),
        ::testing::ValuesIn(framework_)
);

INSTANTIATE_TEST_SUITE_P(smoke_Proposal_tests, ProposalLayerTest,
                        ::testing::Combine(
                                proposalParams,
                                ::testing::ValuesIn({ov::element::f16, ov::element::f32}),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ProposalLayerTest::getTestCaseName);

}  // namespace
