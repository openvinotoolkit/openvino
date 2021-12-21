// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id tile::type_id() {
    static primitive_type_base<tile> instance;
    return &instance;
}

layout tile_inst::calc_output_layout(tile_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for tile_node!");
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto input_format = input_layout.format;
    return layout{input_layout.data_type, input_format, desc->out_shape};
}

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
