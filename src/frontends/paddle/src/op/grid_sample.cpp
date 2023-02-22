// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "reduce_ops.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs grip_sample(const NodeContext& node) {
    auto data = node.get_input('X');
    auto grid = node.get_input('Grid');

    default_opset::GridSample::Attributes attributes{};
    if (node.has_attribute("align_corners")) {
        attributes.align_corners = node.get_attribute_value<int64_t>("align_corners", 0);
    }
    if (node.has_attribute("mode")) {
        attributes.mode = EnumNames<default_opset::GridSample::InterpolationMode>::as_enum(
            node.get_attribute_value<std::string>("mode", "bilinear"));
    }
    if (node.has_attribute("padding_mode")) {
        attributes.padding_mode = EnumNames<default_opset::GridSample::PaddingMode>::as_enum(
            node.get_attribute_value<std::string>("padding_mode", "zeros"));
    }
    
    return {std::make_shared<default_opset::GridSample>(data, grid, attributes)};
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
