// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_topkrois.hpp"

#include "experimental_detectron_topkrois_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"

namespace ov {
op::v6::ExperimentalDetectronTopKROIs::ExperimentalDetectronTopKROIs(const Output<Node>& input_rois,
                                                                     const Output<Node>& rois_probs,
                                                                     size_t max_rois)
    : Op({input_rois, rois_probs}),
      m_max_rois(max_rois) {
    constructor_validate_and_infer_types();
}

bool op::v6::ExperimentalDetectronTopKROIs::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_ExperimentalDetectronTopKROIs_visit_attributes);
    visitor.on_attribute("max_rois", m_max_rois);
    return true;
}

std::shared_ptr<Node> op::v6::ExperimentalDetectronTopKROIs::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ExperimentalDetectronTopKROIs_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v6::ExperimentalDetectronTopKROIs>(new_args.at(0), new_args.at(1), m_max_rois);
}

void op::v6::ExperimentalDetectronTopKROIs::validate_and_infer_types() {
    OV_OP_SCOPE(v6_ExperimentalDetectronTopKROIs_validate_and_infer_types);

    auto out_et = element::dynamic;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(out_et, get_input_element_type(0), get_input_element_type(1)) &&
                              (out_et.is_dynamic() || out_et.is_real()),
                          "ROIs and probabilities of ROIs must same floating-point type.");

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, out_et, output_shapes[0]);
}

void op::v6::ExperimentalDetectronTopKROIs::set_max_rois(size_t max_rois) {
    m_max_rois = max_rois;
}
}  // namespace ov
