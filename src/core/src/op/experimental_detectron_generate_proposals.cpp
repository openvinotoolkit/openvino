// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_generate_proposals.hpp"

#include "experimental_detectron_generate_proposals_shape_inference.hpp"
#include "experimental_detectron_shape_infer_utils.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"

namespace ov {
namespace op {
namespace v6 {

ExperimentalDetectronGenerateProposalsSingleImage::ExperimentalDetectronGenerateProposalsSingleImage(
    const Output<Node>& im_info,
    const Output<Node>& anchors,
    const Output<Node>& deltas,
    const Output<Node>& scores,
    const Attributes& attrs)
    : Op({im_info, anchors, deltas, scores}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> ExperimentalDetectronGenerateProposalsSingleImage::clone_with_new_inputs(
    const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ExperimentalDetectronGenerateProposalsSingleImage>(new_args.at(0),
                                                                               new_args.at(1),
                                                                               new_args.at(2),
                                                                               new_args.at(3),
                                                                               m_attrs);
}

bool ExperimentalDetectronGenerateProposalsSingleImage::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_visit_attributes);
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("nms_threshold", m_attrs.nms_threshold);
    visitor.on_attribute("post_nms_count", m_attrs.post_nms_count);
    visitor.on_attribute("pre_nms_count", m_attrs.pre_nms_count);
    return true;
}

void ExperimentalDetectronGenerateProposalsSingleImage::validate_and_infer_types() {
    OV_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_validate_and_infer_types);

    const auto shapes_and_type = detectron::validate::all_inputs_same_floating_type(this);
    const auto output_shapes = shape_infer(this, shapes_and_type.first);

    set_output_type(0, shapes_and_type.second, output_shapes[0]);
    set_output_type(1, shapes_and_type.second, output_shapes[1]);
}

void ExperimentalDetectronGenerateProposalsSingleImage::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}
}  // namespace v6
}  // namespace op
}  // namespace ov
