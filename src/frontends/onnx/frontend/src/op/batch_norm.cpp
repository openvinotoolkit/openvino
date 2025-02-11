// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_norm.hpp"

#include <cstdint>
#include <memory>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
// This version supports ONNX BatchNormalization-1 and BatchNormalization-6
ov::OutputVector batch_norm(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
    auto x = inputs.at(0);
    auto scale = inputs.at(1);
    auto bias = inputs.at(2);
    ov::Output<ov::Node> mean;
    ov::Output<ov::Node> var;

    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};

    // Currently only BatchNormalization inference mode is supported by OpenVINO
    std::int64_t is_test{node.get_attribute_value<std::int64_t>("is_test", 1)};
    CHECK_VALID_NODE(node, is_test, "only 'is_test' mode is supported.");

    // optional outputs
    auto after_bn_mean = std::make_shared<NullNode>();
    auto after_bn_var = std::make_shared<NullNode>();
    auto saved_mean = std::make_shared<NullNode>();
    auto saved_var = std::make_shared<NullNode>();

    if (inputs.size() >= 5) {
        mean = inputs.at(3);
        var = inputs.at(4);
        return {std::make_shared<v5::BatchNormInference>(x, scale, bias, mean, var, epsilon),
                after_bn_mean,
                after_bn_var,
                saved_mean,
                saved_var};
    }

    OPENVINO_THROW("Cannot create OpenVINO batch norm with unsupported number of inputs");
}
ONNX_OP("BatchNormalization", OPSET_RANGE(1, 6), ai_onnx::opset_1::batch_norm);
}  // namespace opset_1
/*
     Opset 6 is skipped because there are no significant difference between opset1 and opset6.
     Found difference is:
     1. In Training, the computation of ReduceMean and ReduceVar uses float
        to avoid overflow for float16 inputs.
 */

namespace opset_7 {
// This version supports ONNX BatchNormalization-7 and BatchNormalization-9
ov::OutputVector batch_norm(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
    auto x = inputs.at(0);
    auto scale = inputs.at(1);
    auto bias = inputs.at(2);
    auto mean = inputs.at(3);
    auto var = inputs.at(4);

    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};
    // Attribute "spatial" is ignored, as we only support inference mode of
    // BatchNormalization

    CHECK_VALID_NODE(node, node.get_outputs_size() == 1, "Training mode of BatchNormalization is not supported.");

    return {std::make_shared<v5::BatchNormInference>(x, scale, bias, mean, var, epsilon)};
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
    auto mean = inputs.at(3);
    auto var = inputs.at(4);

    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};
    int64_t training_mode{node.get_attribute_value<int64_t>("training_mode", 0)};

    CHECK_VALID_NODE(node,
                     training_mode == false && node.get_outputs_size() == 1,
                     "Training mode of BatchNormalization is not supported.");
    return {std::make_shared<v5::BatchNormInference>(x, scale, bias, mean, var, epsilon)};
}
ONNX_OP("BatchNormalization", OPSET_SINCE(14), ai_onnx::opset_14::batch_norm);
}  // namespace opset_14
/*
     Opset 15 is skipped because there are no significant difference between opset14 and opset15.
     Found difference is:
     1. In Training, the computation of ReduceMean and ReduceVar uses float
        to avoid overflow for float16 inputs.
 */

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
