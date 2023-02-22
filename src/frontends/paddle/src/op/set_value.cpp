// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

#define MAX_VALUE(T) std::numeric_limits<T>::max()

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

std::shared_ptr<default_opset::Constant> get_max_value_by_dtype(ov::element::Type dtype) {
    if (dtype == element::f32)
        return default_opset::Constant::create(dtype, {}, {MAX_VALUE(float)});
    else
        return default_opset::Constant::create(dtype, {}, {MAX_VALUE(int)});
};

NamedOutputs set_value(const NodeContext& node) {
    const auto input_node = node.get_input("Input");
    auto value_node = node.get_input("ValueTensor");
    const auto input_shape = input_node.get_partial_shape().get_shape();
    const auto dims = static_cast<int64_t>(input_node.get_partial_shape().rank().get_length());
    const auto input_value_dim = value_node.get_partial_shape().get_shape().size();
    const auto dtype = input_node.get_element_type();

    PADDLE_OP_CHECK(node, (!node.has_input("StartsTensorList")), "Slice must be interger");
    PADDLE_OP_CHECK(node, (!node.has_input("StepsTensorList")), "Slice must be interger");
    PADDLE_OP_CHECK(node, (!node.has_input("EndsTensorList")), "Slice must be interger");

    const auto axes = node.get_attribute<std::vector<int64_t>>("axes");
    const auto steps = node.get_attribute<std::vector<int64_t>>("steps");
    const auto starts = node.get_attribute<std::vector<int64_t>>("starts");
    const auto ends = node.get_attribute<std::vector<int64_t>>("ends");

    for (size_t i = 0; i < steps.size(); i++)
        PADDLE_OP_CHECK(node, (steps[i] == 1), "Elements of steps must be 1");

    std::vector<int64_t> padding_starts(dims), padding_ends(dims);
    Shape value_shape(input_shape);

    for (size_t i = 0; i < starts.size(); i++) {
        int64_t s = starts[i], e = ends[i], axis = axes[i], dim = input_shape[axis];
        s += s < 0 ? dim : 0;
        e += e < 0 ? dim : 0;
        padding_starts[axis] = s;
        padding_ends[axis] = dim - e;
        value_shape[axis] -= (s + padding_ends[axis]);
    }

    if (input_value_dim == 1) {
        const auto target_shape = default_opset::Constant::create(element::i64, {value_shape.size()}, value_shape);
        value_node = std::make_shared<default_opset::Broadcast>(value_node, target_shape);
    }

    const auto maximum_value = get_max_value_by_dtype(dtype);

    const auto p_start = default_opset::Constant::create(element::i64, {value_shape.size()}, padding_starts);
    const auto p_end = default_opset::Constant::create(element::i64, {value_shape.size()}, padding_ends);

    const auto padded_value = std::make_shared<default_opset::Pad>(value_node,
                                                                   p_start,
                                                                   p_end,
                                                                   maximum_value,
                                                                   ngraph::op::PadMode::CONSTANT);

    const auto value_mask = std::make_shared<default_opset::Equal>(padded_value, maximum_value);

    return node.default_single_output_mapping(
        {std::make_shared<default_opset::Select>(value_mask, input_node, padded_value)},
        {"Out"});
};

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
