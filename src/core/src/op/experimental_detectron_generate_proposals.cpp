// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_generate_proposals.hpp"

#include "experimental_detectron_generate_proposals_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"

using namespace std;
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

shared_ptr<Node> ExperimentalDetectronGenerateProposalsSingleImage::clone_with_new_inputs(
    const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ExperimentalDetectronGenerateProposalsSingleImage>(new_args.at(0),
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

    const auto input_size = get_input_size();
    auto out_et = element::dynamic;

    auto input_shapes = std::vector<ov::PartialShape>();
    input_shapes.reserve(input_size);

    for (size_t i = 0; i < input_size; ++i) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(out_et, out_et, get_input_element_type(i)) &&
                                  (out_et.is_dynamic() || out_et.is_real()),
                              "Input[",
                              i,
                              "] type '",
                              get_input_element_type(i),
                              "' is not floating point or not same as others inputs.");
        input_shapes.push_back(get_input_partial_shape(i));
    }

    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, out_et, output_shapes[0]);
    set_output_type(1, out_et, output_shapes[1]);
}

void ExperimentalDetectronGenerateProposalsSingleImage::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}
}  // namespace v6
}  // namespace op
}  // namespace ov
