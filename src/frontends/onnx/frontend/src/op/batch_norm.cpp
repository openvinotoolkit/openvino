// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_norm.hpp"

#include <cstdint>
#include <memory>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace {
// Axes of all dimensions except the channel one (index 1), used to calculate per-channel statistics.
ov::Output<ov::Node> statistics_axes(const ov::Output<ov::Node>& x) {
    const auto& rank = x.get_partial_shape().rank();
    if (rank.is_static()) {
        std::vector<int64_t> axes{0};
        for (int64_t axis = 2; axis < rank.get_length(); ++axis) {
            axes.push_back(axis);
        }
        return v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
    }

    const auto shape_of_x = std::make_shared<v3::ShapeOf>(x, ov::element::i64);
    const auto rank_of_x = std::make_shared<v0::Squeeze>(std::make_shared<v3::ShapeOf>(shape_of_x, ov::element::i64));
    const auto start = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    const auto step = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    const auto spatial_axes = std::make_shared<v4::Range>(start, rank_of_x, step, ov::element::i64);
    const auto batch_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    return std::make_shared<v0::Concat>(ov::OutputVector{batch_axis, spatial_axes}, 0);
}

struct TrainingBatchNorm {
    ov::Output<ov::Node> y;
    ov::Output<ov::Node> running_mean;
    ov::Output<ov::Node> running_var;
    ov::Output<ov::Node> current_mean;
    ov::Output<ov::Node> current_var;
};

// Normalizes the input with statistics calculated for the current batch and updates the running statistics:
//   running_mean = input_mean * momentum + current_mean * (1 - momentum)
//   running_var = input_var * momentum + current_var * (1 - momentum)
TrainingBatchNorm make_training_batch_norm(const ov::Output<ov::Node>& x,
                                           const ov::Output<ov::Node>& scale,
                                           const ov::Output<ov::Node>& bias,
                                           const ov::Output<ov::Node>& mean,
                                           const ov::Output<ov::Node>& var,
                                           double epsilon,
                                           double momentum) {
    // ONNX requires the batch statistics to be calculated in float to avoid overflow for float16 inputs
    const auto& x_type = x.get_element_type();
    const bool accumulate_in_f32 = x_type.is_real() && x_type.bitwidth() < ov::element::f32.bitwidth();
    const ov::Output<ov::Node> data =
        accumulate_in_f32 ? std::make_shared<v0::Convert>(x, ov::element::f32)->output(0) : x;

    const auto axes = statistics_axes(data);
    const auto mean_keep_dims = std::make_shared<v1::ReduceMean>(data, axes, true);
    const auto deviation = std::make_shared<v1::Subtract>(data, mean_keep_dims);
    const auto var_keep_dims =
        std::make_shared<v1::ReduceMean>(std::make_shared<v1::Multiply>(deviation, deviation), axes, true);

    const auto channels_shape = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    const ov::Output<ov::Node> mean_1d = std::make_shared<v1::Reshape>(mean_keep_dims, channels_shape, false);
    const ov::Output<ov::Node> var_1d = std::make_shared<v1::Reshape>(var_keep_dims, channels_shape, false);

    // BatchNormInference requires all its inputs to have the same element type
    const ov::Output<ov::Node> current_mean =
        accumulate_in_f32 ? std::make_shared<v1::ConvertLike>(mean_1d, x)->output(0) : mean_1d;
    const ov::Output<ov::Node> current_var =
        accumulate_in_f32 ? std::make_shared<v1::ConvertLike>(var_1d, x)->output(0) : var_1d;

    const auto y = std::make_shared<v5::BatchNormInference>(x, scale, bias, current_mean, current_var, epsilon);

    const auto update_running_stat = [&](const ov::Output<ov::Node>& running,
                                         const ov::Output<ov::Node>& current) -> std::shared_ptr<ov::Node> {
        const auto momentum_const = v0::Constant::create(running.get_element_type(), ov::Shape{}, {momentum});
        const auto rest_const = v0::Constant::create(running.get_element_type(), ov::Shape{}, {1.0 - momentum});
        return std::make_shared<v1::Add>(
            std::make_shared<v1::Multiply>(running, momentum_const),
            std::make_shared<v1::Multiply>(std::make_shared<v1::ConvertLike>(current, running), rest_const));
    };
    // the running statistics are updated with the values calculated in the accumulation precision
    const auto running_mean = update_running_stat(mean, mean_1d);
    const auto running_var = update_running_stat(var, var_1d);

    return {y, running_mean, running_var, current_mean, current_var};
}
}  // namespace

namespace opset_1 {
// This version supports ONNX BatchNormalization-1 and BatchNormalization-6
ov::OutputVector batch_norm(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
    auto x = inputs.at(0);
    auto scale = inputs.at(1);
    auto bias = inputs.at(2);

    CHECK_VALID_NODE(node,
                     inputs.size() >= 5,
                     "Cannot create OpenVINO batch norm with unsupported number of inputs: ",
                     inputs.size());
    auto mean = inputs.at(3);
    auto var = inputs.at(4);

    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};
    double momentum{node.get_attribute_value<double>("momentum", 0.9)};

    // 'is_test' equal to 0 means that the operator works in the training mode
    const bool is_test = node.get_attribute_value<std::int64_t>("is_test", 1) != 0;

    if (is_test) {
        return {std::make_shared<v5::BatchNormInference>(x, scale, bias, mean, var, epsilon),
                std::make_shared<NullNode>(),
                std::make_shared<NullNode>(),
                std::make_shared<NullNode>(),
                std::make_shared<NullNode>()};
    }

    const auto bn = make_training_batch_norm(x, scale, bias, mean, var, epsilon, momentum);
    return {bn.y, bn.running_mean, bn.running_var, bn.current_mean, bn.current_var};
}
ONNX_OP("BatchNormalization", OPSET_RANGE(1, 6), ai_onnx::opset_1::batch_norm);
}  // namespace opset_1
/*
     Opset 6 is skipped because there are no significant difference between opset1 and opset6.
     Found difference is:
     1. In Training, the mean and variance reductions use float
        to avoid overflow for float16 inputs.
 */

namespace opset_7 {
// This version supports ONNX BatchNormalization-7 and BatchNormalization-9
ov::OutputVector batch_norm(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
    auto x = inputs.at(0);
    auto scale = inputs.at(1);
    auto bias = inputs.at(2);

    CHECK_VALID_NODE(node,
                     inputs.size() >= 5,
                     "Cannot create OpenVINO batch norm with unsupported number of inputs: ",
                     inputs.size());
    auto mean = inputs.at(3);
    auto var = inputs.at(4);

    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};
    double momentum{node.get_attribute_value<double>("momentum", 0.9)};
    // Attribute "spatial" is ignored, as only the per-channel normalization is supported

    // More than one output means that the operator works in the training mode
    if (node.get_outputs_size() == 1) {
        return {std::make_shared<v5::BatchNormInference>(x, scale, bias, mean, var, epsilon)};
    }

    const auto bn = make_training_batch_norm(x, scale, bias, mean, var, epsilon, momentum);
    return {bn.y, bn.running_mean, bn.running_var, bn.current_mean, bn.current_var};
}
ONNX_OP("BatchNormalization", OPSET_RANGE(7, 13), ai_onnx::opset_7::batch_norm);
}  // namespace opset_7
/*
    Opset 9 is skipped because there are no significant difference between opset7 and opset9.
    Found difference is:
    1. removed -> spatial : int (default is 1)
    If true, compute the mean and variance across per activation. If false, compute the mean and variance across
    per feature over each mini-batch.

 */

namespace opset_14 {
// This version supports ONNX BatchNormalization-14 BatchNormalization-15
ov::OutputVector batch_norm(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
    auto x = inputs.at(0);
    auto scale = inputs.at(1);
    auto bias = inputs.at(2);

    CHECK_VALID_NODE(node,
                     inputs.size() >= 5,
                     "Cannot create OpenVINO batch norm with unsupported number of inputs: ",
                     inputs.size());
    auto mean = inputs.at(3);
    auto var = inputs.at(4);

    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};
    double momentum{node.get_attribute_value<double>("momentum", 0.9)};
    const bool training_mode = node.get_attribute_value<int64_t>("training_mode", 0) != 0;

    CHECK_VALID_NODE(node,
                     training_mode || node.get_outputs_size() == 1,
                     "Number of outputs greater than one requires the training mode to be enabled.");

    if (!training_mode) {
        return {std::make_shared<v5::BatchNormInference>(x, scale, bias, mean, var, epsilon)};
    }

    const auto bn = make_training_batch_norm(x, scale, bias, mean, var, epsilon, momentum);
    if (node.get_outputs_size() == 1) {
        return {bn.y};
    }
    return {bn.y, bn.running_mean, bn.running_var};
}
ONNX_OP("BatchNormalization", OPSET_SINCE(14), ai_onnx::opset_14::batch_norm);
}  // namespace opset_14
/*
     Opset 15 is skipped because there are no significant difference between opset14 and opset15.
     Found difference is:
     1. In Training, the mean and variance reductions use float
        to avoid overflow for float16 inputs.
 */

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
