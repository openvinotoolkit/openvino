/*
// Copyright (c) 2019 Intel Corporation
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

#include "shuffle_channels_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id shuffle_channels::type_id() {
    static primitive_type_base<shuffle_channels> instance;
    return &instance;
}

layout shuffle_channels_inst::calc_output_layout(shuffle_channels_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;

    const int32_t number_of_dims = 4;
    const int32_t group = desc->group;
    int32_t axis = desc->axis;

    if (axis < 0)
        axis += number_of_dims;

    if (axis < 0 || axis >= number_of_dims)
        CLDNN_ERROR_MESSAGE(node.id(), "Incorrect axis value! Actual axis is" + std::to_string(group));

    if (group < 1)
        CLDNN_ERROR_MESSAGE(
            node.id(),
            "Invalid group size value (should equal at least one). Actual block size is" + std::to_string(group));

    if (input_layout.size.sizes(format::bfyx)[axis] % group != 0)
        CLDNN_ERROR_MESSAGE(
            node.id(),
            "Group parameter must evenly divide the channel dimension. Actual group size is " + std::to_string(group));

    return layout{input_layout.data_type, input_format, input_layout.size};
}

std::string shuffle_channels_inst::to_string(shuffle_channels_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite shuffle_channels_info;
    shuffle_channels_info.add("input id", input.id());
    shuffle_channels_info.add("groups number", desc->group);
    shuffle_channels_info.add("axis", desc->axis);

    node_info->add("shuffle_channels info", shuffle_channels_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

shuffle_channels_inst::typed_primitive_inst(network_impl& network, shuffle_channels_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
