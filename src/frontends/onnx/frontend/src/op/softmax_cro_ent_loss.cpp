// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/log_softmax.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
extern ov::OutputVector negative_log_likelihood_loss_impl(const ov::OutputVector inputs,
                                                          const std::string reduction,
                                                          bool use_ignore_index = false,
                                                          const int64_t ignore_index_value = 0);
}  // namespace opset_1
}  // namespace ai_onnx

ov::OutputVector onnx_softmax_crossentropy_loss(const ov::frontend::onnx::Node& node, int64_t axis_default) {
    const auto& inputs = node.get_ov_inputs();
    const auto& logits = inputs.at(0);
    const auto& target = inputs.at(1);

    int64_t axis = node.get_attribute_value<int64_t>("axis", axis_default);
    auto log_probs = std::make_shared<ov::op::v5::LogSoftmax>(logits, axis);

    ov::OutputVector nlll_inputs{log_probs, target};
    if (inputs.size() > 2)
        nlll_inputs.push_back(inputs[2]);

    const std::string reduction = node.get_attribute_value<std::string>("reduction", "mean");
    bool use_ignore_index = node.has_attribute("ignore_index");
    int64_t ignore_index_value = 0;
    if (use_ignore_index)
        ignore_index_value = node.get_attribute_value<int64_t>("ignore_index");

    auto outputs = ai_onnx::opset_1::negative_log_likelihood_loss_impl(nlll_inputs,
                                                                       reduction,
                                                                       use_ignore_index,
                                                                       ignore_index_value);

    if (node.get_outputs_size() == 2)
        outputs.push_back(log_probs);

    return outputs;
}

namespace ai_onnx {
namespace opset_12 {

ov::OutputVector softmax_cross_entropy_loss(const ov::frontend::onnx::Node& node) {
    return onnx_softmax_crossentropy_loss(node, 1);
}

ONNX_OP("SoftmaxCrossEntropyLoss", OPSET_RANGE(1, 12), ai_onnx::opset_12::softmax_cross_entropy_loss);
}  // namespace opset_12

namespace opset_13 {

ov::OutputVector softmax_cross_entropy_loss(const ov::frontend::onnx::Node& node) {
    return onnx_softmax_crossentropy_loss(node, 1);
}

ONNX_OP("SoftmaxCrossEntropyLoss", OPSET_SINCE(13), ai_onnx::opset_13::softmax_cross_entropy_loss);
}  // namespace opset_13
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
