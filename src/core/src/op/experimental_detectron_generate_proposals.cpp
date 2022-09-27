// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/experimental_detectron_generate_proposals.hpp"

#include <experimental_detectron_generate_proposals_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v6::ExperimentalDetectronGenerateProposalsSingleImage);

op::v6::ExperimentalDetectronGenerateProposalsSingleImage::ExperimentalDetectronGenerateProposalsSingleImage(
    const Output<Node>& im_info,
    const Output<Node>& anchors,
    const Output<Node>& deltas,
    const Output<Node>& scores,
    const Attributes& attrs)
    : Op({im_info, anchors, deltas, scores}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v6::ExperimentalDetectronGenerateProposalsSingleImage::clone_with_new_inputs(
    const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(new_args.at(0),
                                                                                  new_args.at(1),
                                                                                  new_args.at(2),
                                                                                  new_args.at(3),
                                                                                  m_attrs);
}

bool op::v6::ExperimentalDetectronGenerateProposalsSingleImage::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_visit_attributes);
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("nms_threshold", m_attrs.nms_threshold);
    visitor.on_attribute("post_nms_count", m_attrs.post_nms_count);
    visitor.on_attribute("pre_nms_count", m_attrs.pre_nms_count);
    return true;
}

void op::v6::ExperimentalDetectronGenerateProposalsSingleImage::validate_and_infer_types() {
    OV_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_validate_and_infer_types);

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}, ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0),
                                                  get_input_partial_shape(1),
                                                  get_input_partial_shape(2),
                                                  get_input_partial_shape(3)};
    shape_infer(this, input_shapes, output_shapes);

    const auto& input_et = get_input_element_type(0);
    set_output_type(0, input_et, output_shapes[0]);
    set_output_type(1, input_et, output_shapes[1]);
}
