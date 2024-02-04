// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/softmax_cross_entropy_loss.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/frontend/exception.hpp"
#include "ov_models/ov_builders/reshape.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace {
std::shared_ptr<ngraph::Node> onnx_softmax_cross_entropy_loss(const Output<ngraph::Node>& logits,
                                                              const Output<ngraph::Node>& labels,
                                                              const int64_t axis) {
    const auto coerced_logits = ov::op::util::flatten(logits, static_cast<int>(axis));
    const auto coerced_labels = ov::op::util::flatten(labels, static_cast<int>(axis));
    
    const auto result = std::make_shared<default_opset::SoftmaxCrossEntropyLoss>(coerced_logits, coerced_labels, 1);
    const auto logits_shape = std::make_shared<default_opset::ShapeOf>(logits);
    const bool special_zero = false;

    return std::make_shared<default_opset::Reshape>(result, logits_shape, special_zero);
}
}  // namespace

namespace op {
namespace set_1 {
OutputVector softmax_cross_entropy_loss(const Node& node) {
    const auto logits = node.get_ng_inputs().at(0);
    const auto labels = node.get_ng_inputs().at(1);
    
    const auto axis = node.get_attribute_value<int64_t>("axis", 1);

    std::shared_ptr<ngraph::Node> result;
    result = onnx_softmax_cross_entropy_loss(logits, labels, axis);

    return {result};
}
}  // namespace set_1

namespace set_11 {
OutputVector softmax_cross_entropy_loss(const Node& node) {
    const auto logits = node.get_ng_inputs().at(0);
    const auto labels = node.get_ng_inputs().at(1);
    
    const auto axis = node.get_attribute_value<int64_t>("axis", 1);

    std::shared_ptr<ngraph::Node> result;
    result = onnx_softmax_cross_entropy_loss(logits, labels, axis);

    return {result};
}
}  // namespace set_11

namespace set_13 {
OutputVector softmax_cross_entropy_loss(const Node& node) {
    const auto logits = node.get_ng_inputs().at(0);
    const auto labels = node.get_ng_inputs().at(1);
    
    const auto axis = node.get_attribute_value<int64_t>("axis", 1);

    std::shared_ptr<ngraph::Node> result;
    result = onnx_softmax_cross_entropy_loss(logits, labels, axis);

    return {result};
}
}  // namespace set_13
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
