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

#include "tensor_iterator_inst.h"

#include "error_handler.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <exception>
#include <algorithm>

namespace cldnn {
primitive_type_id tensor_iterator::type_id() {
    static primitive_type_base<tensor_iterator> instance;
    return &instance;
}

bool check_if_axis_is_set_properly(tensor_iterator_node const & node) {
    // helper
    auto translate_between_bfyx_and_raw_axis = [](int axis) {
        if (axis == 2)
            return 3;
        if (axis == 3)
            return 2;
        return axis;
    };

    std::vector<tensor_iterator::input_mapping> inputs_with_selected_axis;
    for (const auto& input : node.ports_desc.input_ports) {
        if (input.axis >= 0)
            inputs_with_selected_axis.push_back(input);
    }

    // check if all iteration are performed on the same axis
    int common_axis = -1;
    for (const auto& input : inputs_with_selected_axis) {
        if (common_axis == -1)
            common_axis = input.axis;
        else if (common_axis != input.axis)
            return false;
    }

    // check if size of iteration axis is 1
    for (const auto& input : inputs_with_selected_axis) {
        tensor size = node.get_dependency(input.from).get_output_layout().size;
        for (int i = translate_between_bfyx_and_raw_axis(input.axis) - 1; i >= 0; i--) {
            if (size.raw[translate_between_bfyx_and_raw_axis(i)] != 1)
                return false;
        }
    }
    return true;
}

layout tensor_iterator_inst::calc_output_layout(tensor_iterator_node const & node) {
    if (!check_if_axis_is_set_properly(node))
        CLDNN_ERROR_MESSAGE(node.id(), "axis is not set properly");
    // getting number of interations
    int port_to_iterate_throgh = node.ports_desc.find_input_port_with_selected_axis();
    node.iteration_axis = -1;
    if (port_to_iterate_throgh == -1) {
        node.iterations = 1;
    }
    else {
        node.iteration_axis = node.ports_desc.input_ports[port_to_iterate_throgh].axis;
        node.iterations = node.get_dependency(port_to_iterate_throgh).get_output_layout().size.raw[node.iteration_axis];
    }

    //TODO: check inputs
    node.build_body_program();
    auto outputs = node.get_body_program()->get_outputs();
    for (auto output : outputs) {
        layout l = output->get_output_layout();
        output->set_output_layout(l);
    }
    
    assert(node.ports_desc.output_ports.size() == 1);
    primitive_id main_output_id = node.ports_desc.output_ports[0];
    if (node.is_output_working_as_backedge())
        main_output_id += node.backedge_suffix;

    // finds internal output
    auto target = std::find_if(outputs.begin(), outputs.end(), [&](const auto output) {
        return output->id() == main_output_id;
    });

    if (target == outputs.end())
        CLDNN_ERROR_MESSAGE(node.id(), "output not found");

    layout ti_output_layout = (*target)->get_output_layout();
    if (port_to_iterate_throgh != -1)
         ti_output_layout.size.raw[node.iteration_axis] = node.iterations;
    return ti_output_layout;
}

std::string tensor_iterator_inst::to_string(tensor_iterator_node const & node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite ti_info;
    ti_info.add("body", desc->body.get_primitive_ids());

    std::stringstream primitive_description;
    node_info->add("tensor iterator pooling info", ti_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
    
}
tensor_iterator_inst::typed_primitive_inst(network_impl & network, tensor_iterator_node const & node)
    : parent(network, node),
      body_network(node.get_program().get_engine().allocate_network(
                                                     *node.get_body_program(),
                                                     network.get_stream_id(),
                                                     true)) {
    if (!check_if_axis_is_set_properly(node))
        CLDNN_ERROR_MESSAGE(node.id(), "axis is not set properly");
}
}
