// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_detection_output.hpp"

#include <memory>

#include "experimental_detectron_detection_output_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"

using namespace std;

namespace ov {
op::v6::ExperimentalDetectronDetectionOutput::ExperimentalDetectronDetectionOutput(const Output<Node>& input_rois,
                                                                                   const Output<Node>& input_deltas,
                                                                                   const Output<Node>& input_scores,
                                                                                   const Output<Node>& input_im_info,
                                                                                   const Attributes& attrs)
    : Op({input_rois, input_deltas, input_scores, input_im_info}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

bool op::v6::ExperimentalDetectronDetectionOutput::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_ExperimentalDetectronDetectionOutput_visit_attributes);
    visitor.on_attribute("score_threshold", m_attrs.score_threshold);
    visitor.on_attribute("nms_threshold", m_attrs.nms_threshold);
    visitor.on_attribute("max_delta_log_wh", m_attrs.max_delta_log_wh);
    visitor.on_attribute("num_classes", m_attrs.num_classes);
    visitor.on_attribute("post_nms_count", m_attrs.post_nms_count);
    visitor.on_attribute("max_detections_per_image", m_attrs.max_detections_per_image);
    visitor.on_attribute("class_agnostic_box_regression", m_attrs.class_agnostic_box_regression);
    visitor.on_attribute("deltas_weights", m_attrs.deltas_weights);
    return true;
}

void op::v6::ExperimentalDetectronDetectionOutput::validate_and_infer_types() {
    OV_OP_SCOPE(v6_ExperimentalDetectronDetectionOutput_validate_and_infer_types);

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
    set_output_type(1, element::i32, output_shapes[1]);
    set_output_type(2, out_et, output_shapes[2]);
}

shared_ptr<Node> op::v6::ExperimentalDetectronDetectionOutput::clone_with_new_inputs(
    const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ExperimentalDetectronDetectionOutput_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v6::ExperimentalDetectronDetectionOutput>(new_args.at(0),
                                                                     new_args.at(1),
                                                                     new_args.at(2),
                                                                     new_args.at(3),
                                                                     m_attrs);
}

void op::v6::ExperimentalDetectronDetectionOutput::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}
}  // namespace ov
