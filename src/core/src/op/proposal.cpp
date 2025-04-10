// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/proposal.hpp"

#include "itt.hpp"
#include "proposal_shape_inference.hpp"

namespace ov {
op::v0::Proposal::Proposal(const Output<Node>& class_probs,
                           const Output<Node>& bbox_deltas,
                           const Output<Node>& image_shape,
                           const Attributes& attrs)
    : Op({class_probs, bbox_deltas, image_shape}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::v0::Proposal::validate_element_types() {
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_real(),
                          "Proposal layer input class_probs should have floating point type (",
                          get_input_element_type(0),
                          ").");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_real(),
                          "Proposal layer input bbox_deltas should have floating point type (",
                          get_input_element_type(1),
                          ").");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).is_real(),
                          "Proposal layer input image_shape should have floating point type (",
                          get_input_element_type(2),
                          ").");
}

void op::v0::Proposal::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Proposal_validate_and_infer_types);

    validate_element_types();
    const auto intput_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, intput_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<Node> op::v0::Proposal::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Proposal_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Proposal>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
}

bool op::v0::Proposal::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Proposal_visit_attributes);
    visitor.on_attribute("base_size", m_attrs.base_size);
    visitor.on_attribute("pre_nms_topn", m_attrs.pre_nms_topn);
    visitor.on_attribute("post_nms_topn", m_attrs.post_nms_topn);
    visitor.on_attribute("nms_thresh", m_attrs.nms_thresh);
    visitor.on_attribute("feat_stride", m_attrs.feat_stride);
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("ratio", m_attrs.ratio);
    visitor.on_attribute("scale", m_attrs.scale);
    visitor.on_attribute("clip_before_nms", m_attrs.clip_before_nms);
    visitor.on_attribute("clip_after_nms", m_attrs.clip_after_nms);
    visitor.on_attribute("normalize", m_attrs.normalize);
    visitor.on_attribute("box_size_scale", m_attrs.box_size_scale);
    visitor.on_attribute("box_coordinate_scale", m_attrs.box_coordinate_scale);
    visitor.on_attribute("framework", m_attrs.framework);
    return true;
}

void op::v0::Proposal::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}

// --- v4 ---
op::v4::Proposal::Proposal(const Output<Node>& class_probs,
                           const Output<Node>& class_bbox_deltas,
                           const Output<Node>& image_shape,
                           const op::v0::Proposal::Attributes& attrs)
    : v0::Proposal() {
    set_arguments({class_probs, class_bbox_deltas, image_shape});
    set_attrs(attrs);
    constructor_validate_and_infer_types();
}

void op::v4::Proposal::validate_and_infer_types() {
    OV_OP_SCOPE(v4_Proposal_validate_and_infer_types);

    validate_element_types();
    const auto intput_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, intput_shapes);
    const auto& out_et = get_input_element_type(0);

    for (size_t i = 0; i < output_shapes.size(); ++i) {
        set_output_type(i, out_et, output_shapes[i]);
    }
    m_attrs.infer_probs = true;  // Proposal v4 requires default true of infer_probs so output_1 has valid data.
}

std::shared_ptr<Node> op::v4::Proposal::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_Proposal_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Proposal>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
}
}  // namespace ov
