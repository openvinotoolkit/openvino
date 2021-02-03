/*
// Copyright (c) 2020 Intel Corporation
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

#include "scatter_nd_update_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id scatter_nd_update::type_id() {
    static primitive_type_base<scatter_nd_update> instance;
    return &instance;
}

layout scatter_nd_update_inst::calc_output_layout(scatter_nd_update_node const& node) {
    auto desc = node.get_primitive();

    const int32_t axis = desc->axis;
    const size_t input_number_of_dims = node.input(0).get_output_layout().size.sizes().size();

    auto input_layout = node.input(0).get_output_layout();

    auto output_shape = input_layout.size;
    auto input_format = input_layout.format;
    auto output_type = input_layout.data_type;

    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    if (static_cast<size_t>(axis) < 0 || static_cast<size_t>(axis) >= input_number_of_dims)
        CLDNN_ERROR_MESSAGE(node.id(), "Incorrect axis value for ScatterNDUpdate: Axis must be positive and less than the input tensor dimension.");

    return layout{output_type, input_format, output_shape};
}

std::string scatter_nd_update_inst::to_string(scatter_nd_update_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite scatter_nd_update_info;
    scatter_nd_update_info.add("input id", input.id());
    scatter_nd_update_info.add("axis", desc->axis);
    scatter_nd_update_info.add("output shape", node.input(0).get_output_layout().size.to_string());

    node_info->add("scatter_nd_update info", scatter_nd_update_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

scatter_nd_update_inst::typed_primitive_inst(network_impl& network, scatter_nd_update_node const& node) : parent(network, node) {}

}  // namespace cldnn
