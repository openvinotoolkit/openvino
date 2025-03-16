// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "softmax_crossentropy_loss.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace {

// Implements the ONNX SoftmaxCrossEntropyLoss operator as described in version 13.
// Given:
//   - scores: tensor of predicted scores with shape [N, C] or [N, C, D1, D2, …, Dk]
//   - labels: tensor of ground truth labels with shape [N] or [N, D1, D2, …, Dk]
//   - weights (optional): 1D tensor of per-class weights
//   - ignore_index (optional): a label value to be ignored (loss set to zero)
// The operator computes:
//   p = Softmax(scores)
//   y = log(p)
//   loss = -y[c] (or multiplied by weights if provided)
// If the label equals ignore_index, loss is zero.
// Finally, the loss tensor L is optionally reduced (none, sum, or mean).

OutputVector onnx_softmax_crossentropy_loss(const Node& node, int64_t axis_default) {
    // Retrieve input nodes
    const auto inputs = node.get_ov_inputs();
    const auto scores = inputs[0];  // Predicted scores (logits)
    const auto labels = inputs[1];  // Ground truth label indices

    // Determine if class weights are provided (optional input)
    bool has_weights = inputs.size() > 2;
    std::shared_ptr<ov::Node> weights_gather = nullptr;

    // Process ignore_index attribute: labels equal to ignore_index should be ignored (loss = 0)
    bool has_ignore_index = node.has_attribute("ignore_index");
    int64_t ignore_index_val = 0;
    std::shared_ptr<ov::Node> mask = nullptr;
    
    if (has_ignore_index) {
        // Retrieve ignore_index value
        ignore_index_val = node.get_attribute_value<int64_t>("ignore_index");
        // Create constant with ignore_index value
        auto ignore_index_node = v0::Constant::create(labels.get_element_type(), {}, {ignore_index_val});
        // Generate a boolean mask where true means label != ignore_index
        auto neq = std::make_shared<v1::NotEqual>(labels, ignore_index_node);
        // Convert mask to same element type as scores (used later for weighting)
        mask = std::make_shared<v0::Convert>(neq, scores.get_element_type());
    }

    // If weights are provided, gather weights corresponding to each label.
    // Otherwise, if ignore_index is set, use the mask as weights.
    if (has_weights) {
        const auto weights = inputs[2];
        // Gather axis is 0 since weights is a 1D tensor with size = number of classes
        const auto axis_for_weights = v0::Constant::create(element::i64, {}, {0});
        weights_gather = std::make_shared<v8::Gather>(weights, labels, axis_for_weights);
        // Apply mask to zero out weights for ignored labels if needed
        if (has_ignore_index) {
            weights_gather = std::make_shared<v1::Multiply>(weights_gather, mask);
        }
    } else if (has_ignore_index) {
        // No explicit weights provided; use the mask directly.
        weights_gather = mask;
    }

    // Retrieve the 'axis' (channel dimension) and 'reduction' attributes.
    // 'axis' determines which dimension to compute the softmax over.
    // 'reduction' can be "none", "sum", or "mean" (default: "mean")
    const auto axis = node.get_attribute_value<int64_t>("axis", axis_default);
    const auto reduction = node.get_attribute_value<std::string>("reduction", "mean");

    // Compute the log-softmax of the scores along the specified axis.
    // This computes: y = log(softmax(scores))
    const auto log_softmax = std::make_shared<v5::LogSoftmax>(scores, axis);

    std::shared_ptr<ov::Node> gathered;
    std::shared_ptr<ov::Node> loss;

    // The expected behavior is that the loss tensor L has the same shape as the labels tensor.
    // For higher-dimensional inputs (rank > 3) we flatten the trailing dimensions.
    auto log_softmax_shape = log_softmax->get_output_partial_shape(0);
    bool needs_reshape = log_softmax_shape.rank().is_static() && log_softmax_shape.size() > 3;

    if (needs_reshape) {
        // --- Handle N-D input (N, C, D1, D2, ..., Dk) ---
        // Reshape log_softmax to a 3D tensor of shape (N, C, D),
        // where D is the product of all trailing dimensions (D1 * D2 * ... * Dk)
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

        // Reshape labels to a 2D tensor of shape (N, D)
        const auto labels_shape = labels.get_shape();
        int64_t labels_N = labels_shape[0];
        int64_t labels_D = 1;
        for (size_t i = 1; i < labels_shape.size(); ++i) {
            labels_D *= labels_shape[i];
        }
        const auto new_labels_shape = ov::Shape{static_cast<size_t>(labels_N), static_cast<size_t>(labels_D)};
        const auto new_labels_shape_const = v0::Constant::create(ov::element::i64, {2}, new_labels_shape);
        auto reshaped_labels = std::make_shared<v1::Reshape>(labels, new_labels_shape_const, true);

        // Gather the log-softmax values using reshaped labels.
        // For each sample, gather the log probability corresponding to the true class.
        // Note: here, the gather is performed along axis=1 (the channel dimension).
        const auto axis_const = v0::Constant::create(element::i64, {}, {1});
        gathered = std::make_shared<v8::Gather>(reshaped_log_softmax, reshaped_labels, axis_const, 0);

        // Reshape the gathered result back to the original shape of labels (i.e. [N, D1, D2, ...]).
        const auto original_labels_shape = labels.get_shape();
        const auto target_shape_const =
            v0::Constant::create(ov::element::i64, {original_labels_shape.size()}, original_labels_shape);
        auto reshaped_gathered = std::make_shared<v1::Reshape>(gathered, target_shape_const, true);

        // Compute the per-element loss: L = - gathered_log_prob.
        loss = std::make_shared<v0::Negative>(reshaped_gathered);
    } else {
        // --- Handle 2-D input (or rank <= 3) ---
        // Gather the log probabilities directly using labels as indices.
        const auto axis_const = v0::Constant::create(element::i64, {}, {axis});
        gathered = std::make_shared<v8::Gather>(log_softmax, labels, axis_const, 1);
        loss = std::make_shared<v0::Negative>(gathered);
    }

    // If weights are provided (or derived from ignore_index mask), multiply the loss element-wise.
    if (weights_gather) {
        loss = std::make_shared<v1::Multiply>(loss, weights_gather);
    }

    // Apply the reduction if specified.
    // - "none": output shape remains the same as labels.
    // - "sum": output is a scalar (sum over all elements).
    // - "mean": output is a scalar (mean over all elements) or weighted mean if weights are provided.
    if (reduction != "none") {
        auto loss_shape = loss->get_output_partial_shape(0);
        if (loss_shape.rank().is_static()) {
            // Create a constant containing all axes to reduce.
            std::vector<int64_t> reduce_axes(loss_shape.rank().get_length());
            std::iota(reduce_axes.begin(), reduce_axes.end(), 0);
            auto reduce_axis = v0::Constant::create(ov::element::i64, {reduce_axes.size()}, reduce_axes);

            if (reduction == "mean") {
                if (weights_gather) {
                    // Compute weighted mean: sum(loss)/sum(weights)
                    auto loss_sum = std::make_shared<v1::ReduceSum>(loss, reduce_axis, false);
                    auto weight_sum = std::make_shared<v1::ReduceSum>(weights_gather, reduce_axis, false);
                    loss = std::make_shared<v1::Divide>(loss_sum, weight_sum);
                } else {
                    // Compute mean loss over all elements.
                    loss = std::make_shared<v1::ReduceMean>(loss, reduce_axis, false);
                }
            } else if (reduction == "sum") {
                // Sum over all elements.
                loss = std::make_shared<v1::ReduceSum>(loss, reduce_axis, false);
            }
        } else {
            // Dynamic rank is not supported for reduction.
            OPENVINO_THROW("Dynamic rank is not supported for SoftmaxCrossEntropyLoss reduction.");
        }
    }

    // Prepare the output vector.
    // First output is always the computed loss.
    // If the ONNX node expects a second output, return the log_softmax tensor.
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

}  // namespace onnx
}  // namespace frontend
}  // namespace ov