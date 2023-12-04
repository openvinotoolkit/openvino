// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/proposal.hpp"

#include <memory>
#include <vector>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeProposal(const ov::Output<Node>& class_probs,
                                   const ov::Output<Node>& class_logits,
                                   const std::vector<float>& image_info,
                                   const element::Type& type,
                                   size_t base_size,
                                   size_t pre_nms_topn,
                                   size_t post_nms_topn,
                                   float nms_thresh,
                                   size_t feat_stride,
                                   size_t min_size,
                                   const std::vector<float>& ratio,
                                   const std::vector<float>& scale,
                                   bool clip_before_nms,
                                   bool clip_after_nms,
                                   bool normalize,
                                   float box_size_scale,
                                   float box_coordinate_scale,
                                   std::string framework) {
    ov::op::v4::Proposal::Attributes attrs;
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

    auto image_shape = makeConstant(ov::element::Type_t::f32, {3}, image_info);

    return std::make_shared<ov::op::v4::Proposal>(class_probs, class_logits, image_shape, attrs);
}

}  // namespace builder
}  // namespace ngraph
