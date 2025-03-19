// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/detection_output.hpp"

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace org_openvinotoolkit {
namespace opset_1 {
ov::OutputVector detection_output(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();

    auto box_logits = inputs[0];
    auto class_preds = inputs[1];
    auto proposals = inputs[2];

    v8::DetectionOutput::Attributes attrs;
    attrs.background_label_id = static_cast<int>(node.get_attribute_value<int64_t>("background_label_id", 0));
    attrs.top_k = static_cast<int>(node.get_attribute_value<int64_t>("top_k", -1));
    attrs.variance_encoded_in_target = node.get_attribute_value<int64_t>("variance_encoded_in_target", 0);
    // spec says keep_top_k is an array of ints, but some models use a single int
    // also CPU plugin expects single integer
    attrs.keep_top_k = {static_cast<int>(node.get_attribute_value<int64_t>("keep_top_k", 1))};

    auto code_type = node.get_attribute_value<std::string>("code_type", std::string{"caffe.PriorBoxParameter.CORNER"});
    // possible values are "caffe.PriorBoxParameter.CENTER_SIZE",
    // "caffe.PriorBoxParameter.CORNER"
    // but we can just have "CENTER_SIZE" or "CORNER" strings here, so we need to
    // handle that case
    if (code_type.find("caffe.PriorBoxParameter.") == std::string::npos) {
        code_type = "caffe.PriorBoxParameter." + code_type;
    }
    attrs.code_type = code_type;
    attrs.share_location = node.get_attribute_value<int64_t>("share_location", 1);
    attrs.nms_threshold = node.get_attribute_value<float>("nms_threshold");
    attrs.confidence_threshold = node.get_attribute_value<float>("confidence_threshold", 0);
    attrs.clip_after_nms = node.get_attribute_value<int64_t>("clip_after_nms", 0);
    attrs.clip_before_nms = node.get_attribute_value<int64_t>("clip_before_nms", 0);
    attrs.decrease_label_id = node.get_attribute_value<int64_t>("decrease_label_id", 0);
    // TODO: per spec, normalized by default should be 0, but in MO it's 1.
    attrs.normalized = node.get_attribute_value<int64_t>("normalized", 1);
    attrs.input_width = node.get_attribute_value<int64_t>("input_width", 1);
    attrs.input_height = node.get_attribute_value<int64_t>("input_height", 1);
    attrs.objectness_score = node.get_attribute_value<float>("objectness_score", 0);

    if (inputs.size() == 3) {
        return {std::make_shared<v8::DetectionOutput>(box_logits, class_preds, proposals, attrs)};
    } else if (inputs.size() == 5) {
        auto aux_class_preds = inputs[3];
        auto aux_box_preds = inputs[4];
        return {std::make_shared<v8::DetectionOutput>(box_logits,
                                                      class_preds,
                                                      proposals,
                                                      aux_class_preds,
                                                      aux_box_preds,
                                                      attrs)};
    } else {
        FRONT_END_GENERAL_CHECK(false, "Invalid number of inputs");
    }
}

ONNX_OP("DetectionOutput", OPSET_SINCE(1), org_openvinotoolkit::opset_1::detection_output, OPENVINO_ONNX_DOMAIN);
}  // namespace opset_1
}  // namespace org_openvinotoolkit
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
