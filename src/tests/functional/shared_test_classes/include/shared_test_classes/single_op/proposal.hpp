// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using proposalSpecificParams = std::tuple<size_t,              // base_size
                                          size_t,              // pre_nms_topn
                                          size_t,              // post_nms_topn
                                          float,               // nms_thresh
                                          size_t,              // min_size
                                          std::vector<float>,  // ratio
                                          std::vector<float>,  // scale
                                          bool,                // clip_before_nms
                                          bool,                // clip_after_nms
                                          std::string          // framework
                                          >;
using proposalLayerTestParamsSet = std::tuple<proposalSpecificParams, ov::element::Type, ov::test::TargetDevice>;

class ProposalLayerTest : public testing::WithParamInterface<proposalLayerTestParamsSet>,
                          virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<proposalLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
