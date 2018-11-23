/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "tile_inst.h"
#include "primitive_type_base.h"
#include "memory_impl.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id tile_type_id()
{
    static primitive_type_base<tile> instance;
    return &instance;
}

layout tile_inst::calc_output_layout(tile_node const& node)
{
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto input_format = input_layout.format;
    auto result_sizes = input_layout.size.sizes();

    auto axis_index = node.get_primitive()->axis;
    auto tiles = node.get_primitive()->tiles;

    // calculate sum of features from all inputs
    result_sizes[axis_index] *= tiles;
    return layout{ input_layout.data_type, input_format, result_sizes };
}

std::string tile_inst::to_string(tile_node const& node)
{
    auto desc           = node.get_primitive();
    auto node_info      = node.desc_to_json();
    auto& input         = node.input();
    
    std::stringstream primitive_description;

    json_composite tile_info;
    tile_info.add("input id", input.id());
    tile_info.add("axis", desc->axis);
    tile_info.add("tiles", desc->tiles);
    
    node_info->add("tile info", tile_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

tile_inst::typed_primitive_inst(network_impl& network, tile_node const& node)
    :parent(network, node)
{
}

}
