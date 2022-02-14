// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include "openvino/frontend/paddle/node_context.hpp"

#include "default_opset.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

void check_rank(const NodeContext& node, const ngraph::Output<ov::Node>& input, const int64_t input_rank)
{
    std::string input_name = input.get_node()->get_friendly_name();
    PADDLE_OP_CHECK(node, input.get_partial_shape().rank().is_static(), "generate proposals: " + input_name + " rank must be static!");
    PADDLE_OP_CHECK(node, input.get_partial_shape().rank() == Dimension(input_rank), "generate proposals: " + input_name + " rank must be " + std::to_string(input_rank));
}

std::shared_ptr<ngraph::Node> get_one_batch(const NodeContext& node, const ngraph::Output<ov::Node>& input, const int64_t input_rank)
{
    std::string input_name = input.get_node()->get_friendly_name();
    check_rank(node, input, input_rank);
    PADDLE_OP_CHECK(node, input.get_partial_shape()[0].is_static(), "generate proposals: " + input_name + " the first dimension of shape (batch size dimension) must be static!");
    PADDLE_OP_CHECK(node, input.get_partial_shape()[0] == Dimension(1), "generate proposals: " + input_name + " the first dimension of shape (batch size dimension) must be 1!");

    auto squeeze_axes = default_opset::Constant::create<int32_t>(ov::element::i64, {1}, {0});
    auto output = std::make_shared<default_opset::Squeeze>(input, squeeze_axes);

    return output;
}

NamedOutputs generate_proposals_v2(const NodeContext& node)
{
    auto bbox_deltas = node.get_input("BboxDeltas"); // [N，4 * A，H，W]
    auto im_shape = node.get_input("ImShape");       // [N, 2]
    auto scores = node.get_input("Scores");          // [N，A，H，W]
    auto anchors = node.get_input("Anchors");        // [H，W，A，4]
    Output<Node> variances;
    if (node.has_input("Variances"))
        variances = node.get_input("Variances");    // [H，W，A，4]

    auto single_bbox_deltas = get_one_batch(node, bbox_deltas, 4);
    auto single_im_shape = get_one_batch(node, im_shape, 2);
    auto single_scores = get_one_batch(node, scores, 4);

    check_rank(node, anchors, 4);

    // attribute
    ov::op::v8::ExperimentalDetectronGenerateProposalsSingleImage::Attributes attrs;
    attrs.min_size = node.get_attribute<float>("min_size", 0.1);
    attrs.nms_threshold = node.get_attribute<float>("nms_thresh", 0.5);
    attrs.pre_nms_count = node.get_attribute<int>("pre_nms_topN", 6000);
    attrs.post_nms_count = node.get_attribute<int>("post_nms_topN", 1000);
    attrs.dynamic_output = true;
    float eta = node.get_attribute<float>("eta", 1.0);
    PADDLE_OP_CHECK(node, (eta == 1.0), "Only support case of eta == 1.0 currently");
    attrs.coordinates_offset = node.get_attribute<bool>("pixel_offset", true);

    // reshape H, W, A to H * W * A
    auto anchors_shape = default_opset::Constant::create<int64_t>(ov::element::i64, {2}, {-1, 4});
    auto reshaped_anchors = std::make_shared<default_opset::Reshape>(anchors, anchors_shape, true);

    auto variances_bbox_deltas = single_bbox_deltas;
    if (variances.get_node()) {
        // Transpose variances from [H, W, A, 4] to [A*4, H, W]
        auto reshape_pattern = default_opset::Constant::create<int64_t>(ov::element::i64, {3}, {0, 0, -1});
        auto reshaped_variances = std::make_shared<default_opset::Reshape>(variances, reshape_pattern, true);
        auto transpose_order = default_opset::Constant::create(ov::element::i64, {3}, {2, 0, 1});
        auto transposed_variances = std::make_shared<default_opset::Transpose>(reshaped_variances, transpose_order);
        //auto transposed_variances = default_opset::Constant::create(ov::element::f32, {}, {2.0});
        variances_bbox_deltas = std::make_shared<default_opset::Multiply>(single_bbox_deltas, transposed_variances);
    }

    // im_info
    auto im_scale = default_opset::Constant::create(ov::element::f32, {1}, {1.0});
    auto im_info = std::make_shared<default_opset::Concat>(OutputVector{single_im_shape, im_scale}, 0);

    // input:
    //  1. im_info: [H, W, S]
    //  2. anchors: [H*W*A, 4]
    //  3. deltas: [A*4, H, W]
    //  4. scores: [A, H, W]
    // output:
    //  1. rois: [proposals_num, 4]
    //  2. scores: [proposals_num, 1]
    //  2. proposals_num: [1]
    auto proposal = std::make_shared<ov::op::v8::ExperimentalDetectronGenerateProposalsSingleImage>(im_info, reshaped_anchors, variances_bbox_deltas, single_scores, attrs);

    auto unsqueeze_scalar = default_opset::Constant::create(ov::element::i64, {}, {1});
    auto probs = std::make_shared<default_opset::Unsqueeze>(proposal->output(1), unsqueeze_scalar);

    // output
    NamedOutputs named_outputs;
    named_outputs["RpnRois"] = OutputVector{proposal->output(0)};
    named_outputs["RpnRoiProbs"] = OutputVector{probs->output(0)};
    named_outputs["RpnRoisNum"] = OutputVector{proposal->output(2)};

    // TODO: experimental generate proposal testcase
    return named_outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
