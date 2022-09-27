// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/proposal.hpp"

#include <proposal_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::Proposal);

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
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0),
                                                  get_input_partial_shape(1),
                                                  get_input_partial_shape(2)};
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

shared_ptr<Node> op::v0::Proposal::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Proposal_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Proposal>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
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

BWDCMP_RTTI_DEFINITION(op::v4::Proposal);

op::v4::Proposal::Proposal(const Output<Node>& class_probs,
                           const Output<Node>& class_bbox_deltas,
                           const Output<Node>& image_shape,
                           const op::v0::Proposal::Attributes& attrs)
    : v0::Proposal(class_probs, class_bbox_deltas, image_shape, attrs) {
    constructor_validate_and_infer_types();
}

void op::v4::Proposal::validate_and_infer_types() {
    OV_OP_SCOPE(v4_Proposal_validate_and_infer_types);
    v0::Proposal::validate_element_types();

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}, ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0),
                                                  get_input_partial_shape(1),
                                                  get_input_partial_shape(2)};
    shape_infer(this, input_shapes, output_shapes);

    const auto& input0_type = get_input_element_type(0);
    set_output_type(0, input0_type, output_shapes[0]);
    set_output_type(1, input0_type, output_shapes[1]);
}

std::shared_ptr<Node> op::v4::Proposal::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_Proposal_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v4::Proposal>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
}
