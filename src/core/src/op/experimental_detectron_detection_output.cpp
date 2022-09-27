// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/experimental_detectron_detection_output.hpp"

#include <experimental_detectron_detection_output_shape_inference.hpp>
#include <memory>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/host_tensor.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v6::ExperimentalDetectronDetectionOutput);

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

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}, ov::PartialShape{}, ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0),
                                                  get_input_partial_shape(1),
                                                  get_input_partial_shape(2),
                                                  get_input_partial_shape(3)};
    shape_infer(this, input_shapes, output_shapes);

    auto input_et = get_input_element_type(0);

    set_output_size(3);
    set_output_type(0, input_et, output_shapes[0]);
    set_output_type(1, element::Type_t::i32, output_shapes[1]);
    set_output_type(2, input_et, output_shapes[2]);
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
