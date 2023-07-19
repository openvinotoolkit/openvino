// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/grid_sample.hpp"

#include "openvino/opsets/opset9.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector grid_sample(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);
    const auto grid = node.get_ng_inputs().at(1);

    ov::opset9::GridSample::Attributes attributes{};
    attributes.align_corners = node.get_attribute_value<int64_t>("align_corners", 0);

    attributes.mode = EnumNames<ov::opset9::GridSample::InterpolationMode>::as_enum(
        node.get_attribute_value<std::string>("mode", "bilinear"));

    attributes.padding_mode = EnumNames<ov::opset9::GridSample::PaddingMode>::as_enum(
        node.get_attribute_value<std::string>("padding_mode", "zeros"));

    return {std::make_shared<ov::opset9::GridSample>(data, grid, attributes)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
