// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_to_space_inst.h"
#include "depth_to_space_shape_inference.hpp"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(depth_to_space)

template<typename ShapeType>
std::vector<layout> depth_to_space_inst::calc_output_layouts(depth_to_space_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<depth_to_space>();
    auto input_layout = impl_param.get_input_layout(0);
    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);
    auto output_format = input_layout.format;

    ov::op::v0::DepthToSpace op;
    op.set_block_size(desc->block_size);

    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>()
    };
    std::vector<ShapeType> output_shapes = ov::op::v0::shape_infer(&op, input_shapes);

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> depth_to_space_inst::calc_output_layouts<ov::PartialShape>(depth_to_space_node const& node, const kernel_impl_params& impl_param);

std::string depth_to_space_inst::to_string(depth_to_space_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite depth_to_space_info;
    depth_to_space_info.add("input id", input.id());
    depth_to_space_info.add("block size", desc->block_size);
    depth_to_space_info.add("mode", desc->mode == depth_to_space_mode::blocks_first ? "blocks_first" : "depth_first");

    node_info->add("depth_to_space info", depth_to_space_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

depth_to_space_inst::typed_primitive_inst(network& network, depth_to_space_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
