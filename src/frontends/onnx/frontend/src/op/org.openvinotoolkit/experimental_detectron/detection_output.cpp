// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/experimental_detectron_detection_output.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace org_openvinotoolkit {
namespace opset_1 {
ov::OutputVector experimental_detectron_detection_output(const ov::frontend::onnx::Node& node) {
    using DetectionOutput = v6::ExperimentalDetectronDetectionOutput;

    auto inputs = node.get_ov_inputs();
    auto rois = inputs[0];
    auto deltas = inputs[1];
    auto scores = inputs[2];
    auto im_info = inputs[3];

    DetectionOutput::Attributes attrs{};
    attrs.score_threshold = node.get_attribute_value<float>("score_threshold", 0.05f);
    attrs.nms_threshold = node.get_attribute_value<float>("nms_threshold", 0.5f);
    attrs.max_delta_log_wh = node.get_attribute_value<float>("max_delta_log_wh", std::log(1000.0f / 16.0f));
    attrs.num_classes = node.get_attribute_value<std::int64_t>("num_classes", 81);
    attrs.post_nms_count = node.get_attribute_value<std::int64_t>("post_nms_count", 2000);
    attrs.max_detections_per_image = node.get_attribute_value<std::int64_t>("max_detections_per_image", 100);
    attrs.class_agnostic_box_regression =
        static_cast<bool>(node.get_attribute_value<std::int64_t>("class_agnostic_box_regression", 0));
    attrs.deltas_weights = node.get_attribute_value<std::vector<float>>("deltas_weights", {10.0f, 10.0f, 5.0f, 5.0f});
    auto detection_output = std::make_shared<DetectionOutput>(rois, deltas, scores, im_info, attrs);
    return {detection_output->output(0), detection_output->output(1), detection_output->output(2)};
}

ONNX_OP("ExperimentalDetectronDetectionOutput",
        OPSET_SINCE(1),
        org_openvinotoolkit::opset_1::experimental_detectron_detection_output,
        OPENVINO_ONNX_DOMAIN);
}  // namespace opset_1
}  // namespace org_openvinotoolkit
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
