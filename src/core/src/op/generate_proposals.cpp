// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/generate_proposals.hpp"

#include "generate_proposals_shape_inference.hpp"
#include "itt.hpp"

namespace ov {

op::v9::GenerateProposals::GenerateProposals(const Output<Node>& im_info,
                                             const Output<Node>& anchors,
                                             const Output<Node>& deltas,
                                             const Output<Node>& scores,
                                             const Attributes& attrs,
                                             const element::Type& roi_num_type)
    : Op({im_info, anchors, deltas, scores}),
      m_attrs(attrs),
      m_roi_num_type(roi_num_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v9::GenerateProposals::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v9_GenerateProposals_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::op::v9::GenerateProposals>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           m_attrs,
                                                           m_roi_num_type);
}

bool op::v9::GenerateProposals::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v9_GenerateProposals_visit_attributes);
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("nms_threshold", m_attrs.nms_threshold);
    visitor.on_attribute("post_nms_count", m_attrs.post_nms_count);
    visitor.on_attribute("pre_nms_count", m_attrs.pre_nms_count);
    visitor.on_attribute("normalized", m_attrs.normalized);
    visitor.on_attribute("nms_eta", m_attrs.nms_eta);
    visitor.on_attribute("roi_num_type", m_roi_num_type);
    return true;
}

void op::v9::GenerateProposals::validate_and_infer_types() {
    OV_OP_SCOPE(v9_GenerateProposals_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, m_attrs.pre_nms_count > 0, "Attribute pre_nms_count must be larger than 0.");
    NODE_VALIDATION_CHECK(this, m_attrs.post_nms_count > 0, "Attribute post_nms_count must be larger than 0.");
    NODE_VALIDATION_CHECK(this, m_attrs.nms_eta == 1.0, "Attribute min_size must be 1.0.");

    std::vector<PartialShape> input_shapes = {get_input_partial_shape(0),
                                              get_input_partial_shape(1),
                                              get_input_partial_shape(2),
                                              get_input_partial_shape(3)};
    const auto output_shapes = shape_infer(this, input_shapes);

    const auto& input_et = get_input_element_type(0);
    set_output_type(0, input_et, output_shapes[0]);
    set_output_type(1, input_et, output_shapes[1]);
    const auto& roi_num_type = get_roi_num_type();
    NODE_VALIDATION_CHECK(this,
                          (roi_num_type == element::i64) || (roi_num_type == element::i32),
                          "The third output type must be int64 or int32.");
    set_output_type(2, roi_num_type, output_shapes[2]);
}

void op::v9::GenerateProposals::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}

}  // namespace ov
