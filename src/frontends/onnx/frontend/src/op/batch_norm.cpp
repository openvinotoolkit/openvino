// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/batch_norm.hpp"

#include <cstdint>
#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "onnx_import/core/null_node.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
// This version supports ONNX BatchNormalization-1 and BatchNormalization-6
OutputVector batch_norm(const Node& node) {
    OutputVector inputs{node.get_ng_inputs()};
    auto x = inputs.at(0);
    auto scale = inputs.at(1);
    auto bias = inputs.at(2);
    Output<ngraph::Node> mean;
    Output<ngraph::Node> var;

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
        return {std::make_shared<default_opset::BatchNormInference>(x, scale, bias, mean, var, epsilon),
                after_bn_mean,
                after_bn_var,
                saved_mean,
                saved_var};
    }

    OPENVINO_THROW("Cannot create OpenVINO batch norm with unsupported number of inputs");
}
}  // namespace set_1

namespace set_7 {
// This version supports ONNX BatchNormalization-7 and BatchNormalization-9
OutputVector batch_norm(const Node& node) {
    OutputVector inputs{node.get_ng_inputs()};
    auto x = inputs.at(0);
    auto scale = inputs.at(1);
    auto bias = inputs.at(2);
    auto mean = inputs.at(3);
    auto var = inputs.at(4);

    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};
    // Attribute "spatial" is ignored, as we only support inference mode of
    // BatchNormalization

    CHECK_VALID_NODE(node, node.get_outputs_size() == 1, "Training mode of BatchNormalization is not supported.");

    return {std::make_shared<default_opset::BatchNormInference>(x, scale, bias, mean, var, epsilon)};
}

}  // namespace set_7

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
