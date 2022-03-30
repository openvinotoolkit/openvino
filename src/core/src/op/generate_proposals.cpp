// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/generate_proposals.hpp"

#include "generate_proposals_shape_inference.hpp"
#include "itt.hpp"

using namespace std;

ov::op::v9::GenerateProposals::GenerateProposals(const Output<Node>& im_info,
                                                 const Output<Node>& anchors,
                                                 const Output<Node>& deltas,
                                                 const Output<Node>& scores,
                                                 const Attributes& attrs,
                                                 const element::Type roi_num_type)
    : Op({im_info, anchors, deltas, scores}),
      m_attrs(attrs),
      m_roi_num_type(roi_num_type) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> ov::op::v9::GenerateProposals::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v9_GenerateProposals_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ov::op::v9::GenerateProposals>(new_args.at(0),
                                                      new_args.at(1),
                                                      new_args.at(2),
                                                      new_args.at(3),
                                                      m_attrs,
                                                      m_roi_num_type);
}

bool ov::op::v9::GenerateProposals::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v9_GenerateProposals_visit_attributes);
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("nms_threshold", m_attrs.nms_threshold);
    visitor.on_attribute("post_nms_count", m_attrs.post_nms_count);
    visitor.on_attribute("pre_nms_count", m_attrs.pre_nms_count);
    visitor.on_attribute("normalized", m_attrs.normalized);
    visitor.on_attribute("nms_eta", m_attrs.nms_eta);
    visitor.on_attribute("roi_num_type", m_roi_num_type);
    return true;
}

void ov::op::v9::GenerateProposals::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v9_GenerateProposals_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, m_attrs.pre_nms_count > 0, "Attribute pre_nms_count must be larger than 0.");
    NODE_VALIDATION_CHECK(this, m_attrs.post_nms_count > 0, "Attribute post_nms_count must be larger than 0.");
    NODE_VALIDATION_CHECK(this, m_attrs.nms_eta == 1.0, "Attribute min_size must be 1.0.");

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}, ov::PartialShape{}, ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0),
                                                  get_input_partial_shape(1),
                                                  get_input_partial_shape(2),
                                                  get_input_partial_shape(3)};
    shape_infer(this, input_shapes, output_shapes);

    const auto& input_et = get_input_element_type(0);
    set_output_type(0, input_et, output_shapes[0]);
    set_output_type(1, input_et, output_shapes[1]);
    const auto roi_num_type = get_roi_num_type();
    NODE_VALIDATION_CHECK(this,
                          (roi_num_type == ov::element::i64) || (roi_num_type == ov::element::i32),
                          "The third output type must be int64 or int32.");
    set_output_type(2, roi_num_type, output_shapes[2]);
}
