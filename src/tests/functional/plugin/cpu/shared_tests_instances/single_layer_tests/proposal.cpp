// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/proposal.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

/* ============= Proposal ============= */
const std::vector<base_size_type> base_size_ = {16};
const std::vector<pre_nms_topn_type> pre_nms_topn_ = {100};
const std::vector<post_nms_topn_type> post_nms_topn_ = {100};
const std::vector<nms_thresh_type> nms_thresh_ = {0.7f};
const std::vector<min_size_type> min_size_ = {1};
const std::vector<ratio_type> ratio_ = {{1.0f, 2.0f}};
const std::vector<scale_type> scale_ = {{1.2f, 1.5f}};
const std::vector<clip_before_nms_type> clip_before_nms_ = {false};
const std::vector<clip_after_nms_type> clip_after_nms_ = {false};

// empty string corresponds to Caffe framework
const std::vector<framework_type> framework_ = {""};

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
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ProposalLayerTest::getTestCaseName
);
}  // namespace
