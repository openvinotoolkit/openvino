// Copyright (C) 2018-2023 Intel Corporation
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

    const std::unordered_map<std::string, v9::GridSample::InterpolationMode> grid_sample_mode_map{
        {"bilinear", v9::GridSample::InterpolationMode::BILINEAR},
        {"nearest", v9::GridSample::InterpolationMode::NEAREST}};
    const std::unordered_map<std::string, v9::GridSample::PaddingMode> grid_sample_padding_mode_map{
        {"zeros", v9::GridSample::PaddingMode::ZEROS},
        {"border", v9::GridSample::PaddingMode::BORDER},
        {"reflection", v9::GridSample::PaddingMode::REFLECTION}};

    attributes.align_corners = node.get_attribute<bool>("align_corners", true);
    attributes.mode = grid_sample_mode_map.at(node.get_attribute<std::string>("mode", "bilinear"));
    attributes.padding_mode = grid_sample_padding_mode_map.at(node.get_attribute<std::string>("padding_mode", "zeros"));

    return node.default_single_output_mapping({std::make_shared<default_opset::GridSample>(data, grid, attributes)},
                                              {"Output"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
