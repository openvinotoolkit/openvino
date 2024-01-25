// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline OutputVector relu(const Node& node) {
    OutputVector ng_inputs{node.get_ng_inputs()};
    return {std::make_shared<default_opset::Relu>(ng_inputs.at(0))};
}

}  // namespace set_1

namespace set_6 {

OutputVector relu(const Node& node) {
    OutputVector ng_inputs{node.get_ng_inputs()};
    // Common implementation for opset 6
    OutputVector ng_inputs{node.get_ng_inputs()};
    const auto data_type = get_data_type(node);
    return {std::make_shared<default_opset::Relu>(ng_inputs.at(0))};
}

}  // namespace set_6
namespace set_13 {

OutputVector relu(const Node& node) {
    OutputVector ng_inputs{node.get_ng_inputs()};
    // Implementation specific to opset 13
    OutputVector ng_inputs{node.get_ng_inputs()};
    const auto data_type = get_data_type(node);
    return {std::make_shared<default_opset::Relu>(ng_inputs.at(0))};
}

}  // namespace set_13
namespace set_14 {

OutputVector relu(const Node& node) {
    OutputVector ng_inputs{node.get_ng_inputs()};
    // Implementation specific to opset 14
    OutputVector ng_inputs{node.get_ng_inputs()};
    const auto data_type = get_data_type(node);
    return {std::make_shared<default_opset::Relu>(ng_inputs.at(0))};
}

}  // namespace set_14

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
