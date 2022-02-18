// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/org.openvinotoolkit/experimental_detectron/generate_proposals_single_image.hpp"

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

std::shared_ptr<ngraph::Node> get_one_batch(const ngraph::Output<ov::Node>& input)
{
    std::string input_name = input.get_node()->get_friendly_name();

    auto squeeze_axes = default_opset::Constant::create<int32_t>(ov::element::i64, {1}, {0});
    auto output = std::make_shared<default_opset::Squeeze>(input, squeeze_axes);

    return output;
}

OutputVector generate_proposals(const Node& node) {
    const auto inputs = node.get_ng_inputs();
    NGRAPH_CHECK(inputs.size() == 4,
                 "GenerateProposals expects 4 "
                 "inputs, received: ",
                 inputs.size());

    auto scores = inputs[0];
    auto deltas = inputs[1];
    auto im_info = inputs[2];
    auto anchors = inputs[3];

    // get single batch
    auto single_bbox_deltas = get_one_batch(deltas);
    auto single_im_info = get_one_batch(im_info);
    auto single_scores = get_one_batch(scores);

    // use ExperimentalDetectronPriorGridGenerator to extend base anchors to all anchors
    using ExperimentalDetectronPriorGridGenerator = ngraph::op::v6::ExperimentalDetectronPriorGridGenerator;

    ExperimentalDetectronPriorGridGenerator::Attributes prior_grid_attrs{};
    float stride = 1.0 / node.get_attribute_value<float>("spatial_scale", 1.0/16);
    prior_grid_attrs.stride_x = stride;
    prior_grid_attrs.stride_y = stride;
    prior_grid_attrs.h = 0;
    prior_grid_attrs.w = 0;
    prior_grid_attrs.flatten = true;

    auto single_all_anchors =
        std::make_shared<ExperimentalDetectronPriorGridGenerator>(anchors, scores, scores, prior_grid_attrs);

    // generate proposals
    using GenerateProposalsSingleImage = ngraph::op::v9::GenerateProposalsSingleImage;

    GenerateProposalsSingleImage::Attributes generate_proposals_attrs{};
    generate_proposals_attrs.min_size = node.get_attribute_value<float>("min_size", 16);
    generate_proposals_attrs.nms_threshold = node.get_attribute_value<float>("nms_threshold", 0.7);
    generate_proposals_attrs.post_nms_count = node.get_attribute_value<std::int64_t>("post_nms_count", 300);
    generate_proposals_attrs.pre_nms_count = node.get_attribute_value<std::int64_t>("pre_nms_count", 6000);
    generate_proposals_attrs.normalized = not node.get_attribute_value<bool>("legacy_plus_one", true);
    generate_proposals_attrs.nms_eta = 1.0;

    auto generate_proposals_single_image =
        std::make_shared<GenerateProposalsSingleImage>(single_im_info, single_all_anchors, single_bbox_deltas, single_scores, generate_proposals_attrs);
    return {generate_proposals_single_image->output(0), generate_proposals_single_image->output(1)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
