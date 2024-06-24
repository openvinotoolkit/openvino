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
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector add(const ov::frontend::onnx::Node& node) {
    CHECK_VALID_NODE(node,
                     !node.has_attribute("consumed_inputs"),
                     "consumed_inputs legacy attribute of Add op is not supported");
    return common::handle_opset6_binary_op<v1::Add>(node);
}
ONNX_OP("Add", OPSET_RANGE(1, 5), ai_onnx::opset_1::add);
}  // namespace opset_1

namespace opset_6 {
ov::OutputVector add(const ov::frontend::onnx::Node& node) {
    return common::handle_opset6_binary_op<v1::Add>(node);
}
ONNX_OP("Add", OPSET_IN(6), ai_onnx::opset_6::add);
}  // namespace opset_6

namespace opset_7 {
ov::OutputVector add(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<v1::Add>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}
ONNX_OP("Add", OPSET_RANGE(7, 12), ai_onnx::opset_7::add);
}  // namespace opset_7

namespace opset_13 {
ONNX_OP("Add", OPSET_IN(13), ai_onnx::opset_7::add);
}  // namespace opset_13

namespace opset_14 {
ONNX_OP("Add", OPSET_SINCE(14), ai_onnx::opset_7::add);
}  // namespace opset_14

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
