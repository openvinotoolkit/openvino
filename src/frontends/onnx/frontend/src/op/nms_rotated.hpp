// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <numeric>

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "default_opset.hpp"
#include "onnx_import/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/opsets/opset13.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline OutputVector nms_rotated(const Node& node) {
    auto iou_threshold = node.get_attribute_value<float>("iou_threshold");
    auto score_threshold = node.get_attribute_value<float>("score_threshold");
    auto max_output_boxes_per_class =
        default_opset::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto iou_threshold_const = default_opset::Constant::create(element::f32, Shape{}, {iou_threshold});
    auto score_threshold_const = default_opset::Constant::create(element::f32, Shape{}, {score_threshold});

    auto nms = std::make_shared<ov::opset13::NMSRotated>(node.get_ng_inputs().at(0),
                                                         node.get_ng_inputs().at(1),
                                                         max_output_boxes_per_class,
                                                         iou_threshold_const,
                                                         score_threshold_const,
                                                         false);

    return {nms->output(0)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
