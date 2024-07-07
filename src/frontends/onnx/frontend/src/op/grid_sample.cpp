// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grid_sample.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector grid_sample(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    const auto grid = node.get_ov_inputs().at(1);

    v9::GridSample::Attributes attributes{};
    attributes.align_corners = node.get_attribute_value<int64_t>("align_corners", 0);

    attributes.mode = ov::EnumNames<v9::GridSample::InterpolationMode>::as_enum(
        node.get_attribute_value<std::string>("mode", "bilinear"));

    attributes.padding_mode = ov::EnumNames<v9::GridSample::PaddingMode>::as_enum(
        node.get_attribute_value<std::string>("padding_mode", "zeros"));

    return {std::make_shared<v9::GridSample>(data, grid, attributes)};
}
ONNX_OP("GridSample", OPSET_SINCE(1), ai_onnx::opset_1::grid_sample);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
