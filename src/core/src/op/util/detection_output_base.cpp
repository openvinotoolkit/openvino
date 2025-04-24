// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/detection_output_base.hpp"

#include "detection_output_shape_inference.hpp"

ov::op::util::DetectionOutputBase::DetectionOutputBase(const ov::OutputVector& args) : Op(args) {}

void ov::op::util::DetectionOutputBase::validate_base(const DetectionOutputBase::AttributesBase& attrs) {
    NODE_VALIDATION_CHECK(
        this,
        attrs.code_type == "caffe.PriorBoxParameter.CORNER" || attrs.code_type == "caffe.PriorBoxParameter.CENTER_SIZE",
        "code_type must be either \"caffe.PriorBoxParameter.CORNER\" or "
        "\"caffe.PriorBoxParameter.CENTER_SIZE\"");

    auto box_logits_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          box_logits_et.is_real(),
                          "Box logits' data type must be floating point. Got " + box_logits_et.to_string());
    auto class_preds_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          class_preds_et == box_logits_et,
                          "Class predictions' data type must be the same as box logits type (" +
                              box_logits_et.to_string() + "). Got " + class_preds_et.to_string());
    auto proposals_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          proposals_et.is_real(),
                          "Proposals' data type must be floating point. Got " + proposals_et.to_string());

    if (get_input_size() == 5) {
        auto aux_class_preds_et = get_input_element_type(3);
        NODE_VALIDATION_CHECK(this,
                              aux_class_preds_et == class_preds_et,
                              "Additional class predictions' data type must be the same as class "
                              "predictions data type (" +
                                  class_preds_et.to_string() + "). Got " + aux_class_preds_et.to_string());
        auto aux_box_preds_et = get_input_element_type(4);
        NODE_VALIDATION_CHECK(this,
                              aux_box_preds_et == box_logits_et,
                              "Additional box predictions' data type must be the same as box logits data type (" +
                                  box_logits_et.to_string() + "). Got " + aux_box_preds_et.to_string());
    }
}

bool ov::op::util::DetectionOutputBase::visit_attributes_base(AttributeVisitor& visitor,
                                                              DetectionOutputBase::AttributesBase& attrs) {
    visitor.on_attribute("background_label_id", attrs.background_label_id);
    visitor.on_attribute("top_k", attrs.top_k);
    visitor.on_attribute("variance_encoded_in_target", attrs.variance_encoded_in_target);
    visitor.on_attribute("keep_top_k", attrs.keep_top_k);
    visitor.on_attribute("code_type", attrs.code_type);
    visitor.on_attribute("share_location", attrs.share_location);
    visitor.on_attribute("nms_threshold", attrs.nms_threshold);
    visitor.on_attribute("confidence_threshold", attrs.confidence_threshold);
    visitor.on_attribute("clip_after_nms", attrs.clip_after_nms);
    visitor.on_attribute("clip_before_nms", attrs.clip_before_nms);
    visitor.on_attribute("decrease_label_id", attrs.decrease_label_id);
    visitor.on_attribute("normalized", attrs.normalized);
    visitor.on_attribute("input_height", attrs.input_height);
    visitor.on_attribute("input_width", attrs.input_width);
    visitor.on_attribute("objectness_score", attrs.objectness_score);
    return true;
}

ov::Dimension ov::op::util::DetectionOutputBase::compute_num_classes(const AttributesBase& attrs) {
    NODE_VALIDATION_CHECK(this,
                          3 == get_input_size() || get_input_size() == 5,
                          "A number of arguments must be  equal to 3 or equal to 5. Got  ",
                          get_input_size());

    std::vector<ov::PartialShape> input_shapes;
    for (size_t input_idx = 0; input_idx < get_input_size(); input_idx++)
        input_shapes.push_back(get_input_partial_shape(input_idx));
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};

    int64_t num_classes = 0;
    int64_t num_prior_boxes_calculated = 0;
    ov::op::util::compute_num_classes(this, attrs, input_shapes, num_classes, num_prior_boxes_calculated);
    if (num_classes > 0)
        return ov::Dimension{num_classes};
    else
        return ov::Dimension::dynamic();
}
