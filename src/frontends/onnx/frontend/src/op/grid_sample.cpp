// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grid_sample.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace {
ov::OutputVector grid_sample_impl(const ov::frontend::onnx::Node& node,
                                  const std::unordered_map<std::string, v9::GridSample::InterpolationMode>& mode_map,
                                  const std::string& default_mode) {
    const auto data = node.get_ov_inputs().at(0);
    const auto grid = node.get_ov_inputs().at(1);

    const auto& data_rank = data.get_partial_shape().rank();
    CHECK_VALID_NODE(node,
                     data_rank.is_dynamic() || data_rank.get_length() == 4,
                     "GridSample is only supported for 4D input tensors. Got rank: ",
                     data_rank.get_length());

    v9::GridSample::Attributes attributes{};
    attributes.align_corners = node.get_attribute_value<int64_t>("align_corners", 0);

    const auto mode = node.get_attribute_value<std::string>("mode", default_mode);
    const auto mode_it = mode_map.find(mode);
    CHECK_VALID_NODE(node, mode_it != mode_map.end(), "Unsupported GridSample mode: ", mode);
    attributes.mode = mode_it->second;

    attributes.padding_mode = ov::EnumNames<v9::GridSample::PaddingMode>::as_enum(
        node.get_attribute_value<std::string>("padding_mode", "zeros"));

    return {std::make_shared<v9::GridSample>(data, grid, attributes)};
}
}  // namespace

namespace opset_1 {
ov::OutputVector grid_sample(const ov::frontend::onnx::Node& node) {
    static const std::unordered_map<std::string, v9::GridSample::InterpolationMode> mode_map{
        {"bilinear", v9::GridSample::InterpolationMode::BILINEAR},
        {"bicubic", v9::GridSample::InterpolationMode::BICUBIC},
        {"nearest", v9::GridSample::InterpolationMode::NEAREST},
    };
    return grid_sample_impl(node, mode_map, "bilinear");
}
ONNX_OP("GridSample", OPSET_RANGE(1, 19), ai_onnx::opset_1::grid_sample);
}  // namespace opset_1

namespace opset_20 {
ov::OutputVector grid_sample(const ov::frontend::onnx::Node& node) {
    static const std::unordered_map<std::string, v9::GridSample::InterpolationMode> mode_map{
        {"linear", v9::GridSample::InterpolationMode::BILINEAR},
        {"cubic", v9::GridSample::InterpolationMode::BICUBIC},
        {"nearest", v9::GridSample::InterpolationMode::NEAREST},
    };
    return grid_sample_impl(node, mode_map, "linear");
}
ONNX_OP("GridSample", OPSET_SINCE(20), ai_onnx::opset_20::grid_sample);
}  // namespace opset_20
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
