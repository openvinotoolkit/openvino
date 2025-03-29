// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/op/grid_sample.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

using namespace ov::op;

NamedOutputs grid_sampler(const NodeContext& node) {
    auto data = node.get_input("X");
    auto grid = node.get_input("Grid");
    default_opset::GridSample::Attributes attributes{};

    attributes.align_corners = node.get_attribute<bool>("align_corners", 1);
    attributes.mode = ov::EnumNames<default_opset::GridSample::InterpolationMode>::as_enum(
        node.get_attribute<std::string>("mode", "bilinear"));
    attributes.padding_mode = ov::EnumNames<default_opset::GridSample::PaddingMode>::as_enum(
        node.get_attribute<std::string>("padding_mode", "zeros"));

    return node.default_single_output_mapping({std::make_shared<default_opset::GridSample>(data, grid, attributes)},
                                              {"Output"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
