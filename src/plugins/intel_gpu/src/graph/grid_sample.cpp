// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "grid_sample_inst.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include "grid_sample_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(grid_sample)

layout grid_sample_inst::calc_output_layout(const grid_sample_node& node, const kernel_impl_params& impl_param) {
    const auto data_layout = impl_param.get_input_layout();
    const auto data_sizes = data_layout.get_dims();
    const auto& N = data_sizes[0];
    const auto& C = data_sizes[1];

    const auto grid_layout = impl_param.get_input_layout(1);
    const auto grid_sizes = grid_layout.get_dims();
    const auto& H = grid_sizes[1];
    const auto& W = grid_sizes[2];

    return {data_layout.data_type, data_layout.format, tensor(data_layout.format, {N, C, H, W})};
}

template<typename ShapeType>
std::vector<layout> grid_sample_inst::calc_output_layouts(grid_sample_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto prim = impl_param.typed_desc<grid_sample>();
    auto input0_layout = impl_param.get_input_layout(0);
    auto input1_layout = impl_param.get_input_layout(1);

    ov::op::v9::GridSample op;
    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        input1_layout.get<ShapeType>()
    };

    std::vector<ShapeType> output_shapes = ov::op::v9::shape_infer(&op, input_shapes);

    return { layout{output_shapes[0], input0_layout.data_type, input0_layout.format} };
}

template std::vector<layout> grid_sample_inst::calc_output_layouts<ov::PartialShape>(grid_sample_node const& node, const kernel_impl_params& impl_param);

std::string grid_sample_inst::to_string(const grid_sample_node& node) {
    auto primitive = node.get_primitive();
    json_composite grid_sample_info;
    grid_sample_info.add("data id", node.input().id());
    grid_sample_info.add("grid id", node.input(1).id());
    grid_sample_info.add("align_corners", primitive->attributes.align_corners);
    grid_sample_info.add("mode", ov::as_string(primitive->attributes.mode));
    grid_sample_info.add("padding_mode", ov::as_string(primitive->attributes.padding_mode));

    auto node_info = node.desc_to_json();
    node_info->add("grid_sample info", grid_sample_info);

    std::ostringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

grid_sample_inst::typed_primitive_inst(network& network, grid_sample_node const& node) : parent(network, node) {}
}  // namespace cldnn
