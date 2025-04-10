// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/invalid_cases/proposal.hpp"

namespace ov {
namespace test {

const bool normalize = true;
const size_t feat_stride = 1;
const float box_size_scale = 2.0f;
const float box_coordinate_scale = 2.0f;

std::string ProposalBehTest::getTestCaseName(testing::TestParamInfo<proposalBehTestParamsSet> obj) {
    proposalSpecificParams proposal_params;
    std::string target_device;
    std::vector<float> img_info;
    std::tie(proposal_params, img_info, target_device) = obj.param;

    size_t base_size, pre_nms_topn, post_nms_topn, min_size;
    float nms_thresh;
    std::vector<float> ratio, scale;
    bool clip_before_nms, clip_after_nms;
    std::string framework;

    std::tie(base_size,
             pre_nms_topn,
             post_nms_topn,
             nms_thresh,
             min_size,
             ratio,
             scale,
             clip_before_nms,
             clip_after_nms,
             framework) = proposal_params;

    std::ostringstream result;
    result << "base_size=" << base_size << "_";
    result << "pre_nms_topn=" << pre_nms_topn << "_";
    result << "post_nms_topn=" << post_nms_topn << "_";
    result << "nms_thresh=" << nms_thresh << "_";
    result << "feat_stride=" << feat_stride << "_";
    result << "min_size=" << min_size << "_";
    result << "ratio = " << ov::test::utils::vec2str(ratio) << "_";
    result << "scale = " << ov::test::utils::vec2str(scale) << "_";
    result << "clip_before_nms=" << clip_before_nms << "_";
    result << "clip_after_nms=" << clip_after_nms << "_";
    result << "targetDevice=" << target_device;
    result << "img_info = " << ov::test::utils::vec2str(img_info) << "_";
    result << "framework=" << framework << "_";
    return result.str();
}

void ProposalBehTest::SetUp() {
    proposalSpecificParams proposalParams;
    std::vector<float> img_info;

    std::tie(proposalParams, img_info, targetDevice) = this->GetParam();

    size_t base_size, pre_nms_topn, post_nms_topn, min_size;
    float nms_thresh;
    std::vector<float> ratio, scale;
    bool clip_before_nms, clip_after_nms;
    std::string framework;

    std::tie(base_size,
             pre_nms_topn,
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

    ov::Shape scores_shape = {1, 2 * num_anchors, bottom_h, bottom_w};
    ov::Shape boxes_shape = {1, 4 * num_anchors, bottom_h, bottom_w};

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f16, scores_shape),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f16, boxes_shape)};

    ov::op::v0::Proposal::Attributes attrs;
    attrs.base_size = base_size;
    attrs.pre_nms_topn = pre_nms_topn;
    attrs.post_nms_topn = post_nms_topn;
    attrs.nms_thresh = nms_thresh;
    attrs.feat_stride = feat_stride;
    attrs.min_size = min_size;
    attrs.ratio = ratio;
    attrs.scale = scale;
    attrs.clip_before_nms = clip_before_nms;
    attrs.clip_after_nms = clip_after_nms;
    attrs.normalize = normalize;
    attrs.box_size_scale = box_size_scale;
    attrs.box_coordinate_scale = box_coordinate_scale;
    attrs.framework = framework;
    attrs.infer_probs = true;

    auto image_shape = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{3}, img_info);

    auto proposal = std::make_shared<ov::op::v4::Proposal>(params[0], params[1], image_shape, attrs);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(proposal)};
    function = std::make_shared<ov::Model>(results, params, "proposal");
}
void ProposalBehTest::run() {
    std::vector<ov::Shape> input_shapes;
    for (const auto& param : function->get_parameters())
        input_shapes.emplace_back(param->get_shape());
    init_input_shapes(ov::test::static_shapes_to_test_representation(input_shapes));

    compile_model();
    for (const auto& targetStaticShapeVec : targetStaticShapes) {
        generate_inputs(targetStaticShapeVec);
        validate();
    }
}

TEST_P(ProposalBehTest, Inference) {
    ASSERT_THROW(run(), ov::Exception);
}
}  // namespace test
}  // namespace ov
