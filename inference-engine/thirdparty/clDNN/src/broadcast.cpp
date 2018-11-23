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


#include "broadcast_inst.h"

#include "error_handler.h"
#include "json_object.h"
#include "primitive_type_base.h"


namespace cldnn
{
primitive_type_id broadcast_type_id()
{
    static primitive_type_base<broadcast> instance;
    return &instance;
}

layout broadcast_inst::calc_output_layout(broadcast_node const& node)
{
    auto input_layout = node.input().get_output_layout();
    auto desc         = node.get_primitive();

    auto&& new_size = tensor::max(desc->broadcast_sizes, input_layout.size);
    return {input_layout.data_type, input_layout.format, new_size};
}

std::string broadcast_inst::to_string(broadcast_node const& node)
{
    auto desc = node.get_primitive();

    const auto& broadcast_sizes     = desc->broadcast_sizes;

    auto node_info  = node.desc_to_json();
   
    json_composite broadcast_info;
    broadcast_info.add("broadcast sizes", broadcast_sizes.to_string());

    node_info->add("broadcast info", broadcast_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

broadcast_inst::typed_primitive_inst(network_impl& network, broadcast_node const& node)
    : parent(network, node)
{
    auto input_layout = node.input().get_output_layout();

    const auto input_format = input_layout.format;
    const auto& input_sizes = input_layout.size;

    auto bc_sizes = argument.broadcast_sizes;

    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Input format", input_format.value, "supported broadcast primitive input formats",
                                  format::bfyx, format::yxfb, format::byxf);


    // Check if sizes of broadcast are in proper range.
    CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(), "Broadcast sizes", bc_sizes, "0 value", {1, 1, 1, 1},
                                       "Invalid broadcast size: non-positive value");

    bc_sizes = tensor::max(bc_sizes, input_sizes);

    // Check if sizes of broadcast are compatible with sizes of input.
    CLDNN_ERROR_TENSOR_SIZES_NOT_DIVIDABLE(node.id(), "Broadcast sizes", bc_sizes, "input sizes", input_sizes,
                                           "Invalid broadcast size: not dividable by input size");
}
}