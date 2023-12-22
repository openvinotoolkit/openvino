// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/elu.hpp"

#include <memory>
#include <vector>

#include "default_opset.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector elu(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    double alpha = node.get_attribute_value<double>("alpha", 1);

    return OutputVector{std::make_shared<default_opset::Elu>(data, alpha)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
