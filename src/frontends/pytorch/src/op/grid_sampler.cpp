// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/grid_sample.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::opset10;

OutputVector translate_grid_sampler(NodeContext& context) {
    auto x = context.get_input(0);
    auto grid = context.get_input(1);
    GridSample::Attributes attrs{};
    const std::unordered_map<int64_t, GridSample::InterpolationMode> grid_sample_mode_map{
        {0, GridSample::InterpolationMode::BILINEAR},
        {1, GridSample::InterpolationMode::NEAREST},
        {2, GridSample::InterpolationMode::BICUBIC},
    };
    const std::unordered_map<int64_t, GridSample::PaddingMode> grid_sample_padding_mode_map{
        {0, GridSample::PaddingMode::ZEROS},
        {1, GridSample::PaddingMode::BORDER},
        {2, GridSample::PaddingMode::REFLECTION}};
    auto mode = context.const_input<int64_t>(2);
    FRONT_END_OP_CONVERSION_CHECK(grid_sample_mode_map.count(mode), "Unknown interpolation mode: ", mode);
    attrs.mode = grid_sample_mode_map.at(mode);
    auto padding_mode = context.const_input<int64_t>(3);
    FRONT_END_OP_CONVERSION_CHECK(grid_sample_padding_mode_map.count(padding_mode),
                                  "Unknown padding mode: ",
                                  padding_mode);
    attrs.padding_mode = grid_sample_padding_mode_map.at(padding_mode);
    bool align_corners = false;
    if (!context.input_is_none(4)) {
        align_corners = context.const_input<int64_t>(4);
    }
    attrs.align_corners = align_corners;

    return {context.mark_node(std::make_shared<GridSample>(x, grid, attrs))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov