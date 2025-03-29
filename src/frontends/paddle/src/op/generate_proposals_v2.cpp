// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

NamedOutputs generate_proposals_v2(const NodeContext& node) {
    auto bbox_deltas = node.get_input("BboxDeltas");  // [N，4 * A，H，W]
    auto im_shape = node.get_input("ImShape");        // [N, 2]
    auto scores = node.get_input("Scores");           // [N，A，H，W]
    auto anchors = node.get_input("Anchors");         // [H，W，A，4] or [H * W * A, 4]
    Output<Node> variances;
    if (node.has_input("Variances"))
        variances = node.get_input("Variances");  // [H，W，A，4] or [H * W * A, 4]

    // attribute
    ov::op::v9::GenerateProposals::Attributes attrs;
    float min_size = node.get_attribute<float>("min_size", 0.1f);
    attrs.min_size = min_size < 1.0f ? 1.0f : min_size;
    attrs.nms_threshold = node.get_attribute<float>("nms_thresh", 0.5f);
    attrs.pre_nms_count = node.get_attribute<int>("pre_nms_topN", 6000);
    attrs.post_nms_count = node.get_attribute<int>("post_nms_topN", 1000);
    attrs.nms_eta = node.get_attribute<float>("eta", 1.0f);
    PADDLE_OP_CHECK(node, (attrs.nms_eta == 1.0), "Only support case of eta == 1.0 currently");
    attrs.normalized = !node.get_attribute<bool>("pixel_offset", true);

    // reshape anchors from to [H, W, A, 4] if it is [H * W * A, 4]
    auto scores_shape = std::make_shared<default_opset::ShapeOf>(scores);
    auto gather_indices = default_opset::Constant::create<int64_t>(ov::element::i64, {3}, {2, 3, 1});
    auto gather_axis = default_opset::Constant::create<int64_t>(ov::element::i64, {}, {0});
    auto partial_anchors_shape = std::make_shared<default_opset::Gather>(scores_shape, gather_indices, gather_axis);
    auto const_4 = default_opset::Constant::create<int64_t>(ov::element::i64, {1}, {4});
    auto anchors_shape = std::make_shared<default_opset::Concat>(OutputVector{partial_anchors_shape, const_4}, 0);
    auto reshaped_anchors = std::make_shared<default_opset::Reshape>(anchors, anchors_shape, true);

    auto variances_bbox_deltas = bbox_deltas;
    if (variances.get_node()) {
        // Reshape variances to [H, W, A, 4] if it is [H * W * A, 4]
        auto dim4_variances = std::make_shared<default_opset::Reshape>(variances, anchors_shape, true);
        // Transpose variances from [H, W, A, 4] to [A*4, H, W]
        auto reshape_pattern = default_opset::Constant::create<int64_t>(ov::element::i64, {3}, {0, 0, -1});
        auto reshaped_variances = std::make_shared<default_opset::Reshape>(dim4_variances, reshape_pattern, true);
        auto transpose_order = default_opset::Constant::create(ov::element::i64, {3}, {2, 0, 1});
        auto transposed_variances = std::make_shared<default_opset::Transpose>(reshaped_variances, transpose_order);
        // auto transposed_variances = default_opset::Constant::create(ov::element::f32, {}, {2.0});
        variances_bbox_deltas = std::make_shared<default_opset::Multiply>(bbox_deltas, transposed_variances);
    }

    // generate im_info from im_scale
    auto pads_begin = default_opset::Constant::create<int64_t>(ov::element::i64, {2}, {0, 0});
    auto pads_end = default_opset::Constant::create<int64_t>(ov::element::i64, {2}, {0, 1});
    auto im_scale = default_opset::Constant::create(ov::element::f32, {}, {1.0});
    auto im_info =
        std::make_shared<default_opset::Pad>(im_shape, pads_begin, pads_end, im_scale, ov::op::PadMode::CONSTANT);

    // input:
    //  1. im_info: [N, 3] or [N, 4]
    //  2. anchors: [H, W, A, 4]
    //  3. deltas: [N, A*4, H, W]
    //  4. scores: [N, A, H, W]
    // output:
    //  1. rois: [proposals_num, 4]
    //  2. scores: [proposals_num]
    //  3. roi_num: [N]
    auto proposal = std::make_shared<ov::op::v9::GenerateProposals>(im_info,
                                                                    reshaped_anchors,
                                                                    variances_bbox_deltas,
                                                                    scores,
                                                                    attrs);
    proposal->set_roi_num_type(ov::element::i32);

    auto unsqueeze_scalar = default_opset::Constant::create(ov::element::i64, {}, {1});
    auto probs = std::make_shared<default_opset::Unsqueeze>(proposal->output(1), unsqueeze_scalar);

    // output
    NamedOutputs named_outputs;
    named_outputs["RpnRois"] = OutputVector{proposal->output(0)};
    named_outputs["RpnRoiProbs"] = OutputVector{probs->output(0)};
    named_outputs["RpnRoisNum"] = OutputVector{proposal->output(2)};

    return named_outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
