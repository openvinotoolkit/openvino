// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeProposal(const ngraph::Output<Node> &class_probs,
                                   const ngraph::Output<Node> &class_logits,
                                   const ngraph::Output<Node> &image_shape,
                                   const element::Type &type,
                                   size_t base_size,
                                   size_t pre_nms_topn,
                                   size_t post_nms_topn,
                                   float nms_thresh,
                                   size_t feat_stride,
                                   size_t min_size,
                                   const std::vector<float> &ratio,
                                   const std::vector<float> &scale,
                                   bool clip_before_nms,
                                   bool clip_after_nms,
                                   bool normalize,
                                   float box_size_scale,
                                   float box_coordinate_scale,
                                   std::string framework) {
    ngraph::op::ProposalAttrs attrs;
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

    return std::make_shared<opset1::Proposal>(class_probs, class_logits, image_shape, attrs);
}

}  // namespace builder
}  // namespace ngraph
