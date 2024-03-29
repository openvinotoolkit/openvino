// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/pad.hpp"

#include "exceptions.hpp"
#include "op/pad.hpp"
#include "openvino/op/squeeze.hpp"

using namespace ov::op;

namespace ngraph {
namespace onnx_import {
namespace op {
namespace custom {
namespace set_1 {
ov::OutputVector pad(const Node& node) {
    // auto inputs = node.get_ng_inputs();
    // std::string pad_mode = node.get_attribute_value<std::string>("mode", "constant");
    // node.get_ng_inputs()[1] = std::make_shared<v0::Squeeze>(node.get_ng_inputs()[1]);
    // auto result = set_11::pad(node).at(0);
    auto node_1 = node;
    node_1.get_ng_inputs()[1] = std::make_shared<v0::Squeeze>(node_1.get_ng_inputs()[1]);
    auto result = set_11::pad(node_1).at(0);

    return {result};
}
}  // namespace set_1
}  // namespace custom
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
