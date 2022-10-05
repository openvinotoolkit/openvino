// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_inst.h"
#include "tile_shape_inference.hpp"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/format.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id tile::type_id() {
    static primitive_type_base<tile> instance;
    return &instance;
}

layout tile_inst::calc_output_layout(tile_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_type) == false &&
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
    std::vector<ShapeType> output_shapes = {ShapeType{}};
    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        repeats_shape
    };

    auto& constant_mem = impl_param.memory_deps;
    if (constant_mem.count(1)) {
        auto repeats_mem = constant_mem.at(1);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> repeats_lock(repeats_mem, impl_param.prog.get_stream());
        std::map<size_t, ngraph::HostTensorPtr> const_data = {
            {1, make_host_tensor(repeats_mem->get_layout(), repeats_lock.data())}
        };
        ov::op::v0::shape_infer(&op, input_shapes, output_shapes, const_data);
    } else {
        auto repeats_data = desc->repeats;
        auto repeats_tensor = make_host_tensor({repeats_shape, data_types::i64, format::bfyx}, static_cast<void*>(repeats_data.data()));
        std::map<size_t, ngraph::HostTensorPtr> const_data = {
            {1, repeats_tensor}
        };
        ov::op::v0::shape_infer(&op, input_shapes, output_shapes, const_data);
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
