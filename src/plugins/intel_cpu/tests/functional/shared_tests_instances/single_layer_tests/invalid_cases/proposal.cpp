// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/invalid_cases/proposal.hpp"

#include <vector>

namespace {
using ov::test::ProposalBehTest;

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
const std::vector<std::vector<float>> img_info_invalid = {{0.f, 225.f, 1.f},
                                                          {225.f, -1.f, 1.f},
                                                          {225.f, NAN, 1.f},
                                                          {INFINITY, 100.f, 1.f},
                                                          {225.f, 100.f, NAN},
                                                          {225.f, 100.f, INFINITY}};

// empty string corresponds to Caffe framework
const std::vector<std::string> framework_ = {""};

const auto proposalParams = ::testing::Combine(::testing::ValuesIn(base_size_),
                                               ::testing::ValuesIn(pre_nms_topn_),
                                               ::testing::ValuesIn(post_nms_topn_),
                                               ::testing::ValuesIn(nms_thresh_),
                                               ::testing::ValuesIn(min_size_),
                                               ::testing::ValuesIn(ratio_),
                                               ::testing::ValuesIn(scale_),
                                               ::testing::ValuesIn(clip_before_nms_),
                                               ::testing::ValuesIn(clip_after_nms_),
                                               ::testing::ValuesIn(framework_));

INSTANTIATE_TEST_SUITE_P(invalid,
                         ProposalBehTest,
                         ::testing::Combine(proposalParams,
                                            ::testing::ValuesIn(img_info_invalid),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ProposalBehTest::getTestCaseName);

}  // namespace
