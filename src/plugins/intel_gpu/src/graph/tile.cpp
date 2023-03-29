// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_inst.h"
#include "tile_shape_inference.hpp"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/format.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(tile)

layout tile_inst::calc_output_layout(tile_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for tile_node!");
    auto desc = impl_param.typed_desc<tile>();

    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;

    std::vector<int64_t> repeats = desc->repeats;

    auto out_shape = input_layout.get_dims();
    for (size_t i = 0; i < repeats.size(); ++i) {
        out_shape[i] *= repeats[i];
    }
    return layout{input_layout.data_type, input_format, tensor(input_format, out_shape)};
}

template<typename ShapeType>
std::vector<layout> tile_inst::calc_output_layouts(tile_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<tile>();
    auto input0_layout = impl_param.get_input_layout(0);

    auto output_type = input0_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    ShapeType repeats_shape = impl_param.input_layouts.size() == 2 ? impl_param.get_input_layout(1).get<ShapeType>()
                                                                   : ov::Shape{ desc->repeats.size() };
    ov::op::v0::Tile op;
    std::vector<ShapeType> output_shapes;
    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        repeats_shape
    };

    auto& constant_mem = impl_param.memory_deps;
    if (constant_mem.count(1)) {
        auto repeats_mem = constant_mem.at(1);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> repeats_lock(repeats_mem, impl_param.prog->get_stream());
        const auto& layout = repeats_mem->get_layout();
        const auto repeats_tensor =
            ov::Tensor(data_type_to_element_type(layout.data_type), layout.get_shape(), repeats_lock.data());
        output_shapes = ov::op::v0::shape_infer(&op, input_shapes, {{1, repeats_tensor}});
    } else {
        auto repeats_data = desc->repeats;
        const auto repeats_tensor =
            ov::Tensor(data_type_to_element_type(data_types::i64), repeats_shape.to_shape(), repeats_data.data());
        output_shapes = ov::op::v0::shape_infer(&op, input_shapes, {{1, repeats_tensor}});
    }

    format output_format = format::adjust_to_rank(input0_layout.format, output_shapes[0].size());

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> tile_inst::calc_output_layouts<ov::PartialShape>(tile_node const& node, const kernel_impl_params& impl_param);

std::string tile_inst::to_string(tile_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite tile_info;
    tile_info.add("input id", input.id());
    node_info->add("tile info", tile_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

tile_inst::typed_primitive_inst(network& network, tile_node const& node) : parent(network, node) {}

}  // namespace cldnn
