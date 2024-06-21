// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector add(const ov::frontend::onnx::Node& node) {
    CHECK_VALID_NODE(node,
                     !node.has_attribute("consumed_inputs"),
                     "consumed_inputs legacy attribute of Add op is not supported");
    return common::handle_opset6_binary_op<v1::Add>(node);
}
static bool registered = register_translator("Add", VersionRange{1, 5}, add);
}  // namespace set_1

namespace set_6 {
ov::OutputVector add(const ov::frontend::onnx::Node& node) {
    return common::handle_opset6_binary_op<v1::Add>(node);
}
static bool registered = register_translator("Add", VersionRange::in(6), add);
}  // namespace set_6

namespace set_7 {
ov::OutputVector add(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<v1::Add>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}
static bool registered = register_translator("Add", VersionRange{7, 12}, add);
}  // namespace set_7

namespace set_13 {
static bool registered = register_translator("Add", VersionRange::in(13), op::set_7::add);
}  // namespace set_13

namespace set_14 {
static bool registered = register_translator("Add", VersionRange::since(14), op::set_7::add);
}  // namespace set_14

}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
