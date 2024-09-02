// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/nms_rotated.hpp"

#include <numeric>

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector nms_rotated(const ov::frontend::onnx::Node& node) {
    auto iou_threshold = node.get_attribute_value<float>("iou_threshold");
    auto score_threshold = node.get_attribute_value<float>("score_threshold");
    auto max_output_boxes_per_class =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto iou_threshold_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {iou_threshold});
    auto score_threshold_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {score_threshold});

    auto nms = std::make_shared<ov::op::v13::NMSRotated>(node.get_ov_inputs().at(0),
                                                         node.get_ov_inputs().at(1),
                                                         max_output_boxes_per_class,
                                                         iou_threshold_const,
                                                         score_threshold_const,
                                                         false);

    return {nms->output(0)};
}
ONNX_OP("NMSRotated", OPSET_SINCE(1), ai_onnx::opset_1::nms_rotated, MMDEPLOY_DOMAIN);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
