// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/add.hpp"

#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "utils/common.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
ov::OutputVector add(const Node& node) {
    CHECK_VALID_NODE(node,
                     !node.has_attribute("consumed_inputs"),
                     "consumed_inputs legacy attribute of Add op is not supported");
    return common::handle_opset6_binary_op<v1::Add>(node);
}
}  // namespace set_1

namespace set_6 {
ov::OutputVector add(const Node& node) {
    return common::handle_opset6_binary_op<v1::Add>(node);
}
}  // namespace set_6

namespace set_7 {
ov::OutputVector add(const Node& node) {
    return {std::make_shared<v1::Add>(node.get_ng_inputs().at(0), node.get_ng_inputs().at(1))};
}
}  // namespace set_7

namespace set_13 {
OutputVector add(const Node& node) {
    return common::handle_opset13_binary_op<default_opset::Add>(node);
}
}  // namespace set_13

namespace set_14 {
OutputVector add(const Node& node) {
    return common::handle_opset14_binary_op<default_opset::Add>(node);
}
}  // namespace set_14

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
