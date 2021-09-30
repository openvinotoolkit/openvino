// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/detection_output.hpp"

#include <detection_output_shape_inference.hpp>

#include "itt.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::v0::DetectionOutput);

ov::op::v0::DetectionOutput::DetectionOutput(const Output<Node>& box_logits,
                                             const Output<Node>& class_preds,
                                             const Output<Node>& proposals,
                                             const Output<Node>& aux_class_preds,
                                             const Output<Node>& aux_box_preds,
                                             const Attributes& attrs)
    : Op({box_logits, class_preds, proposals, aux_class_preds, aux_box_preds}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

ov::op::v0::DetectionOutput::DetectionOutput(const Output<Node>& box_logits,
                                             const Output<Node>& class_preds,
                                             const Output<Node>& proposals,
                                             const Attributes& attrs)
    : Op({box_logits, class_preds, proposals}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void ov::op::v0::DetectionOutput::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v0_DetectionOutput_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, m_attrs.num_classes > 0, "Number of classes must be greater than zero");

    NODE_VALIDATION_CHECK(this, m_attrs.keep_top_k.size() > 0, "keep_top_k attribute must be provided");

    NODE_VALIDATION_CHECK(this,
                          m_attrs.code_type == "caffe.PriorBoxParameter.CORNER" ||
                              m_attrs.code_type == "caffe.PriorBoxParameter.CENTER_SIZE",
                          "code_type must be either \"caffe.PriorBoxParameter.CORNER\" or "
                          "\"caffe.PriorBoxParameter.CENTER_SIZE\"");

    auto box_logits_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          box_logits_et.is_real(),
                          "Box logits' data type must be floating point. Got " + box_logits_et.get_type_name());
    auto class_preds_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          class_preds_et == box_logits_et,
                          "Class predictions' data type must be the same as box logits type (" +
                              box_logits_et.get_type_name() + "). Got " + class_preds_et.get_type_name());
    auto proposals_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          proposals_et.is_real(),
                          "Proposals' data type must be floating point. Got " + proposals_et.get_type_name());

    if (get_input_size() == 5) {
        auto aux_class_preds_et = get_input_element_type(3);
        NODE_VALIDATION_CHECK(this,
                              aux_class_preds_et == class_preds_et,
                              "Additional class predictions' data type must be the same as class "
                              "predictions data type (" +
                                  class_preds_et.get_type_name() + "). Got " + aux_class_preds_et.get_type_name());
        auto aux_box_preds_et = get_input_element_type(4);
        NODE_VALIDATION_CHECK(this,
                              aux_box_preds_et == box_logits_et,
                              "Additional box predictions' data type must be the same as box logits data type (" +
                                  box_logits_et.get_type_name() + "). Got " + aux_box_preds_et.get_type_name());
    }

    std::vector<ov::PartialShape> input_shapes;
    for (auto input_idx = 0; input_idx < get_input_size(); input_idx++)
        input_shapes.push_back(get_input_partial_shape(input_idx));
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};

    shape_infer(this, input_shapes, output_shapes);

    set_output_size(1);
    set_output_type(0, box_logits_et, output_shapes[0]);
}

shared_ptr<ov::Node> ov::op::v0::DetectionOutput::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v0_DetectionOutput_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    auto num_args = new_args.size();

    NODE_VALIDATION_CHECK(this, num_args == 3 || num_args == 5, "DetectionOutput accepts 3 or 5 inputs.");

    if (num_args == 3) {
        return make_shared<DetectionOutput>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    } else {
        return make_shared<DetectionOutput>(new_args.at(0),
                                            new_args.at(1),
                                            new_args.at(2),
                                            new_args.at(3),
                                            new_args.at(4),
                                            m_attrs);
    }
}

bool ov::op::v0::DetectionOutput::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v0_DetectionOutput_visit_attributes);
    visitor.on_attribute("num_classes", m_attrs.num_classes);
    visitor.on_attribute("background_label_id", m_attrs.background_label_id);
    visitor.on_attribute("top_k", m_attrs.top_k);
    visitor.on_attribute("variance_encoded_in_target", m_attrs.variance_encoded_in_target);
    visitor.on_attribute("keep_top_k", m_attrs.keep_top_k);
    visitor.on_attribute("code_type", m_attrs.code_type);
    visitor.on_attribute("share_location", m_attrs.share_location);
    visitor.on_attribute("nms_threshold", m_attrs.nms_threshold);
    visitor.on_attribute("confidence_threshold", m_attrs.confidence_threshold);
    visitor.on_attribute("clip_after_nms", m_attrs.clip_after_nms);
    visitor.on_attribute("clip_before_nms", m_attrs.clip_before_nms);
    visitor.on_attribute("decrease_label_id", m_attrs.decrease_label_id);
    visitor.on_attribute("normalized", m_attrs.normalized);
    visitor.on_attribute("input_height", m_attrs.input_height);
    visitor.on_attribute("input_width", m_attrs.input_width);
    visitor.on_attribute("objectness_score", m_attrs.objectness_score);
    return true;
}
