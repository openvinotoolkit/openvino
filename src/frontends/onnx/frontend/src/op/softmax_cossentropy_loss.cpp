// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "softmax_crossentropy_loss.hpp"

using namespace ov::op;

namespace ov::frontend::onnx {
namespace {

OutputVector onnx_softmax_crossentropy_loss(const Node& node, int64_t axis_default) {
    const auto inputs = node.get_ov_inputs();
    const auto scores = inputs[0];
    const auto labels = inputs[1];

    bool has_weights = inputs.size() > 2;
    std::shared_ptr<ov::Node> weights_gather = nullptr;

    bool has_ignore_index = node.has_attribute("ignore_index");
    int64_t ignore_index_val = 0;
    std::shared_ptr<ov::Node> mask = nullptr;

    if (has_ignore_index) {
        ignore_index_val = node.get_attribute_value<int64_t>("ignore_index");
        auto ignore_index_node = v0::Constant::create(labels.get_element_type(), {}, {ignore_index_val});
        auto neq = std::make_shared<v1::NotEqual>(labels, ignore_index_node);
        mask = std::make_shared<v0::Convert>(neq, scores.get_element_type());
    }

    if (has_weights) {
        const auto weights = inputs[2];
        const auto axis_for_weights = v0::Constant::create(element::i64, {}, {0});
        weights_gather = std::make_shared<v8::Gather>(weights, labels, axis_for_weights);
        if (has_ignore_index) {
            weights_gather = std::make_shared<v1::Multiply>(weights_gather, mask);
        }
    } else if (has_ignore_index) {
        weights_gather = mask;
    }

    const auto axis = node.get_attribute_value<int64_t>("axis", axis_default);
    const auto reduction = node.get_attribute_value<std::string>("reduction", "mean");

    const auto log_softmax = std::make_shared<v5::LogSoftmax>(scores, axis);

    std::shared_ptr<ov::Node> gathered;
    std::shared_ptr<ov::Node> loss;

    auto log_softmax_shape = log_softmax->get_output_partial_shape(0);
    bool needs_reshape = log_softmax_shape.rank().is_static() && log_softmax_shape.size() > 3;

    if (needs_reshape) {
        const auto log_softmax_shape_vec = log_softmax->get_shape();
        const int64_t N = log_softmax_shape_vec[0];
        const int64_t C = log_softmax_shape_vec[1];
        int64_t D = 1;
        for (size_t i = 2; i < log_softmax_shape_vec.size(); ++i) {
            D *= log_softmax_shape_vec[i];
        }
        const auto new_shape_log_softmax =
            ov::Shape{static_cast<size_t>(N), static_cast<size_t>(C), static_cast<size_t>(D)};
        const auto new_shape_log_softmax_const = v0::Constant::create(ov::element::i64, {3}, new_shape_log_softmax);
        auto reshaped_log_softmax = std::make_shared<v1::Reshape>(log_softmax, new_shape_log_softmax_const, true);

        const auto labels_shape = labels.get_shape();
        int64_t labels_N = labels_shape[0];
        int64_t labels_D = 1;
        for (size_t i = 1; i < labels_shape.size(); ++i) {
            labels_D *= labels_shape[i];
        }
        const auto new_labels_shape = ov::Shape{static_cast<size_t>(labels_N), 1, static_cast<size_t>(labels_D)};
        const auto new_labels_shape_const = v0::Constant::create(ov::element::i64, {3}, new_labels_shape);
        auto reshaped_labels = std::make_shared<v1::Reshape>(labels, new_labels_shape_const, true);

        gathered = std::make_shared<v6::GatherElements>(reshaped_log_softmax, reshaped_labels, 1);

        const auto original_labels_shape = labels.get_shape();
        const auto target_shape_const =
            v0::Constant::create(ov::element::i64, {original_labels_shape.size()}, original_labels_shape);
        auto reshaped_gathered = std::make_shared<v1::Reshape>(gathered, target_shape_const, true);

        loss = std::make_shared<v0::Negative>(reshaped_gathered);
    } else {
        const auto axis_const = v0::Constant::create(element::i64, {}, {axis});
        gathered = std::make_shared<v8::Gather>(log_softmax, labels, axis_const, 1);
        loss = std::make_shared<v0::Negative>(gathered);
    }

    if (weights_gather) {
        loss = std::make_shared<v1::Multiply>(loss, weights_gather);
    }

    if (reduction != "none") {
        auto loss_shape = loss->get_output_partial_shape(0);
        if (loss_shape.rank().is_static()) {
            std::vector<int64_t> reduce_axes(loss_shape.rank().get_length());
            std::iota(reduce_axes.begin(), reduce_axes.end(), 0);
            auto reduce_axis = v0::Constant::create(ov::element::i64, {reduce_axes.size()}, reduce_axes);

            if (reduction == "mean") {
                if (weights_gather) {
                    auto loss_sum = std::make_shared<v1::ReduceSum>(loss, reduce_axis, false);
                    auto weight_sum = std::make_shared<v1::ReduceSum>(weights_gather, reduce_axis, false);
                    loss = std::make_shared<v1::Divide>(loss_sum, weight_sum);
                } else {
                    loss = std::make_shared<v1::ReduceMean>(loss, reduce_axis, false);
                }
            } else if (reduction == "sum") {
                loss = std::make_shared<v1::ReduceSum>(loss, reduce_axis, false);
            }
        } else {
            OPENVINO_THROW("Dynamic rank is not supported for SoftmaxCrossEntropyLoss reduction.");
        }
    }

    OutputVector outputs;
    outputs.push_back(loss);
    if (node.get_outputs_size() == 2) {
        outputs.push_back(log_softmax);
    }
    return outputs;
}
}  // namespace
namespace ai_onnx {
namespace opset_12 {
OutputVector softmax_cross_entropy_loss(const Node& node) {
    return onnx_softmax_crossentropy_loss(node, 1);
}
ONNX_OP("SoftmaxCrossEntropyLoss", OPSET_RANGE(1, 12), ai_onnx::opset_12::softmax_cross_entropy_loss);
}  // namespace opset_12

namespace opset_13 {
OutputVector softmax_cross_entropy_loss(const Node& node) {
    return onnx_softmax_crossentropy_loss(node, 1);
}
ONNX_OP("SoftmaxCrossEntropyLoss", OPSET_SINCE(13), ai_onnx::opset_13::softmax_cross_entropy_loss);
}  // namespace opset_13
}  // namespace ai_onnx
}  // namespace ov::frontend::onnx