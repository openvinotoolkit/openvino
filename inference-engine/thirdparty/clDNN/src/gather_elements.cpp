/*
// Copyright (c) 2021 Intel Corporation
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

#include "gather_elements_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id gather_elements::type_id() {
    static primitive_type_base<gather_elements> instance;
    return &instance;
}

layout gather_elements_inst::calc_output_layout(gather_elements_node const& node) {
    auto op = node.get_primitive();

    auto input_layout_origin = node.input(0).get_output_layout();
    auto indices_layout_origin = node.input(1).get_output_layout();

    auto input_layout = input_layout_origin.size.sizes(input_layout_origin.format);
    auto indices_layout = indices_layout_origin.size.sizes(indices_layout_origin.format);

    if (node.has_fused_primitives()) {
        input_layout_origin.data_type = node.get_fused_output_layout().data_type;
    }

    // const size_t input_dims = input_layout.size();
    auto output_type = indices_layout_origin.data_type;
    auto output_format = op->output_format;
    auto output_shape = op->output_shape;

    // const auto axis = op->axis;

    // calculate initial output shape
    return layout(output_type, output_format, output_shape);
}

std::string gather_elements_inst::to_string(gather_elements_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_elements_info;
    gather_elements_info.add("input id", input.id());
    gather_elements_info.add("input shape", node.input(0).get_output_layout().size.to_string());
    gather_elements_info.add("indices shape", node.input(1).get_output_layout().size.to_string());
    gather_elements_info.add("output format", calc_output_layout(node).format);
    gather_elements_info.add("output shape", calc_output_layout(node).size.to_string());
    gather_elements_info.add("axis", desc->axis);

    node_info->add("gather_elements info", gather_elements_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_elements_inst::typed_primitive_inst(network_impl& network, gather_elements_node const& node) : parent(network, node) {}

}  // namespace cldnn
