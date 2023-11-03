// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/invalid_cases/proposal.hpp"

using namespace BehaviorTestsDefinitions;
using namespace LayerTestsDefinitions;

std::string ProposalBehTest::getTestCaseName(testing::TestParamInfo<proposalBehTestParamsSet> obj) {
    proposalSpecificParams proposalParams;
    std::string targetDevice;
    std::vector<float> img_info;
    std::tie(proposalParams, img_info, targetDevice) = obj.param;
    auto proposalPramString = ProposalLayerTest::SerializeProposalSpecificParams(proposalParams);

    std::ostringstream result;
    result << "targetDevice=" << targetDevice;
    result << "img_info = " << ov::test::utils::vec2str(img_info) << "_";

    return proposalPramString + result.str();
}

InferenceEngine::Blob::Ptr ProposalBehTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::Blob::Ptr blobPtr;

    const std::string name = info.name();
    if (name == "scores") {
        blobPtr = FuncTestUtils::createAndFillBlobFloat(info.getTensorDesc(), 1, 0, 1000, 8234231);
    } else if (name == "boxes") {
        blobPtr = FuncTestUtils::createAndFillBlobFloatNormalDistribution(info.getTensorDesc(), 0.0f, 0.2f, 7235346);
    }

    return blobPtr;
}

void ProposalBehTest::SetUp() {
    proposalSpecificParams proposalParams;
    std::vector<float> img_info;

    std::tie(proposalParams, img_info, targetDevice) = this->GetParam();
    base_size_type base_size;
    pre_nms_topn_type pre_nms_topn;
    post_nms_topn_type post_nms_topn;
    nms_thresh_type nms_thresh;
    min_size_type min_size;
    ratio_type ratio;
    scale_type scale;
    clip_before_nms_type clip_before_nms;
    clip_after_nms_type clip_after_nms;
    framework_type framework;

    std::tie(base_size, pre_nms_topn,
             post_nms_topn,
             nms_thresh,
             min_size,
             ratio,
             scale,
             clip_before_nms,
             clip_after_nms,
             framework) = proposalParams;

    size_t bottom_w = base_size;
    size_t bottom_h = base_size;
    size_t num_anchors = ratio.size() * scale.size();

    std::vector<size_t> scoresShape = {1, 2 * num_anchors, bottom_h, bottom_w};
    std::vector<size_t> boxesShape  = {1, 4 * num_anchors, bottom_h, bottom_w};
    std::vector<size_t> imageInfoShape = {3};

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP16);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(scoresShape)),
                               std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(boxesShape))};
    params[0]->set_friendly_name("scores");
    params[1]->set_friendly_name("boxes");

    auto proposal = std::dynamic_pointer_cast<ngraph::opset1::Proposal>(
             ngraph::builder::makeProposal(params[0], params[1], img_info, ngPrc,
                                           base_size,
                                           pre_nms_topn,
                                           post_nms_topn,
                                           nms_thresh,
                                           feat_stride,
                                           min_size,
                                           ratio,
                                           scale,
                                           clip_before_nms,
                                           clip_after_nms,
                                           normalize,
                                           box_size_scale,
                                           box_coordinate_scale,
                                           framework));

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(proposal)};
    function = std::make_shared<ngraph::Function>(results, params, "proposal");
}

void ProposalBehTest::Run() {
    LoadNetwork();
    GenerateInputs();
    Infer();
}

TEST_P(ProposalBehTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ASSERT_THROW(Run(), InferenceEngine::Exception);
}
