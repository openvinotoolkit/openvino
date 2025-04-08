// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/proposal.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::ProposalLayerTest;

namespace {

/* ============= Proposal ============= */
const std::vector<size_t> base_size = {16};
const std::vector<size_t> pre_nms_topn = {100};
const std::vector<size_t> post_nms_topn = {100};
const std::vector<float> nms_thresh = {0.7f};
const std::vector<size_t> min_size = {1};
const std::vector<std::vector<float>> ratio = {{1.0f, 2.0f}};
const std::vector<std::vector<float>> scale = {{1.2f, 1.5f}};
const std::vector<bool> clip_before_nms = {false};
const std::vector<bool> clip_after_nms = {false};

// empty string corresponds to Caffe framework
const std::vector<std::string> framework = {""};

const auto proposal_params = ::testing::Combine(
        ::testing::ValuesIn(base_size),
        ::testing::ValuesIn(pre_nms_topn),
        ::testing::ValuesIn(post_nms_topn),
        ::testing::ValuesIn(nms_thresh),
        ::testing::ValuesIn(min_size),
        ::testing::ValuesIn(ratio),
        ::testing::ValuesIn(scale),
        ::testing::ValuesIn(clip_before_nms),
        ::testing::ValuesIn(clip_after_nms),
        ::testing::ValuesIn(framework)
);

INSTANTIATE_TEST_SUITE_P(proposal_params, ProposalLayerTest,
                        ::testing::Combine(
                                proposal_params,
                                ::testing::Values(ov::element::f16),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ProposalLayerTest::getTestCaseName
);
}  // namespace
