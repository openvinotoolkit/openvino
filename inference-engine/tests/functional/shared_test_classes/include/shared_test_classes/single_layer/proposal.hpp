// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

namespace proposalTypes {

typedef size_t base_size_type;
typedef size_t pre_nms_topn_type;
typedef size_t post_nms_topn_type;
typedef float nms_thresh_type;
typedef size_t min_size_type;
typedef std::vector<float> ratio_type;
typedef std::vector<float> scale_type;
typedef bool clip_before_nms_type;
typedef bool clip_after_nms_type;
typedef bool normalize_type;
typedef size_t feat_stride_type;
typedef float box_size_scale_type;
typedef float box_coordinate_scale_type;
typedef std::string framework_type;

};  // namespace proposalTypes

using namespace proposalTypes;

typedef std::tuple<
        base_size_type,
        pre_nms_topn_type,
        post_nms_topn_type,
        nms_thresh_type,
        min_size_type,
        ratio_type,
        scale_type,
        clip_before_nms_type,
        clip_after_nms_type,
        framework_type> proposalSpecificParams;
typedef std::tuple<
        proposalSpecificParams,
        std::string> proposalLayerTestParamsSet;

class ProposalLayerTest
        : public testing::WithParamInterface<proposalLayerTestParamsSet>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<proposalLayerTestParamsSet> obj);
    static std::string SerializeProposalSpecificParams(proposalSpecificParams& params);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
    void Validate() override;
};

}  // namespace LayerTestsDefinitions
