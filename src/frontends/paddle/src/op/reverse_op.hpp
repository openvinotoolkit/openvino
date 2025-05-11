// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
namespace {
NamedOutputs reverse_op(const NodeContext& node) {
    const auto data_node = node.get_input("X");
    const auto axes = node.get_attribute<std::vector<int32_t>>("axis");
    auto axes_length = axes.size();
    const auto starts =
        default_opset::Constant::create(element::i32,
                                        {axes_length},
                                        std::vector<int32_t>(axes_length, std::numeric_limits<int32_t>::max()));
    const auto stops =
        default_opset::Constant::create(element::i32,
                                        {axes_length},
                                        std::vector<int32_t>(axes_length, std::numeric_limits<int32_t>::min()));
    const auto steps =
        default_opset::Constant::create(element::i32, {axes_length}, std::vector<int32_t>(axes_length, -1));
    const auto axes_node = default_opset::Constant::create(element::i32, {axes_length}, axes);

    return node.default_single_output_mapping(
        {std::make_shared<default_opset::Slice>(data_node, starts, stops, steps, axes_node)},
        {"Out"});
}
}  // namespace
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
