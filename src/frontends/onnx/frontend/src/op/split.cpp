// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/split.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/variadic_split.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector split(const ov::frontend::onnx::Node& node) {
    const auto input = node.get_ov_inputs().at(0);
    const auto axis = node.get_attribute_value<int64_t>("axis", 0);

    if (node.has_attribute("split")) {
        const auto splits = node.get_attribute_value<std::vector<int64_t>>("split");
        return ov::op::util::make_split(input, splits, axis);
    } else {
        const auto outputs_number = node.get_output_names().size();
        return ov::op::util::make_split(input, outputs_number, axis);
    }
}

ONNX_OP("Split", OPSET_RANGE(1, 12), ai_onnx::opset_1::split);
}  // namespace opset_1

namespace opset_13 {
ov::OutputVector split(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    const auto axis = node.get_attribute_value<int64_t>("axis", 0);

    if (inputs.size() < 2) {
        const auto outputs_number = node.get_output_names().size();
        return ov::op::util::make_split(inputs.at(0), outputs_number, axis);
    } else {
        const auto axis_node = v0::Constant::create(ov::element::Type_t::i64, ov::Shape{}, {axis});
        return {std::make_shared<v1::VariadicSplit>(inputs.at(0), axis_node, inputs.at(1))->outputs()};
    }
}

ONNX_OP("Split", OPSET_SINCE(13), ai_onnx::opset_13::split);
}  // namespace opset_13
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
