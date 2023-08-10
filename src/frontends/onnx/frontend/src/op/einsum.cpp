// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/einsum.hpp"

#include "default_opset.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector einsum(const Node& node) {
    const std::string& equation{node.get_attribute_value<std::string>("equation")};

    return OutputVector{std::make_shared<default_opset::Einsum>(node.get_ng_inputs(), equation)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
