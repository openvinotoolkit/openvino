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

#include "loop_inst.h"

#include "error_handler.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "api/data.hpp"
#include "api/mutable_data.hpp"
#include <string>
#include <exception>
#include <algorithm>

// TODO(cldnn loop): clDNN/src/loop.cpp calc_output_layout
//   - [x] loop_inst::calc_output_layout
//   - [x] loop_inst::to_string
//   - [x] loop_inst::typed_primitive_inst
namespace cldnn {
primitive_type_id loop::type_id() {
    static primitive_type_base<loop> instance;
    return &instance;
}

static bool check_if_axis_is_set_properly(loop_node const & node) {
    const auto& input_primitive_map = node.get_input_primitive_map();

    std::vector<std::reference_wrapper<const loop::primitive_mapping>> input_with_axis_iteration;
    for (const auto& input : input_primitive_map) {
        if (input.axis >= 0) {
            input_with_axis_iteration.push_back(std::cref(input));
        }
    }

    // check all iteration axis has the same size
    const std::vector<cldnn::program_node *>& dependencies = node.get_dependencies();
    int32_t iteration_size = -1;
    for (const auto& pm : input_with_axis_iteration) {
        auto found = std::find_if(dependencies.begin(), dependencies.end(), [&pm](const cldnn::program_node * node){
            return node->id() == pm.get().external_id;
        });
        assert(found != dependencies.end());
        const layout input_layout = (*found)->get_output_layout();
        const auto shape = input_layout.size.sizes(input_layout.format);
        const int32_t iteration_axis = node.convert_to_raw_axis(pm.get().axis, shape.size());
        if (iteration_size < 0) {
            iteration_size = shape[iteration_axis];
        } else {
            if (iteration_size != shape[iteration_axis]) {
                return false;
            }
        }
    }

    // check if size of iteration axis is 1
    for (const auto& input_ref : input_with_axis_iteration) {
        const loop::primitive_mapping& input = input_ref.get();
        auto dep = std::find_if(dependencies.begin(), dependencies.end(),
            [&input](const cldnn::program_node *dep) { return input.external_id == dep->id(); });

        // if corresponding external id is not found
        if (dep == dependencies.end()) {
            return false;
        }

        // TODO(cldnn loop): need this check? all axis size should be 1 except iteration axis
        // tensor size = node.get_dependency(input.from).get_output_layout().size;
        // for (int i = translate_between_bfyx_and_raw_axis(input.axis) - 1; i >= 0; i--) {
        //     if (size.raw[translate_between_bfyx_and_raw_axis(i)] != 1)
        //         return false;
        // }
    }
    return true;
}

static void validate_backedges(loop_node const & node) {
    const auto& back_edges = node.get_back_edges();
    const auto& input_primitive_map = node.get_input_primitive_map();

    // check input with iteration axis has backedge
    for (const auto& back_edge : back_edges) {
        for (const auto& mapping : input_primitive_map) {
            if (mapping.internal_id == back_edge.to && mapping.axis >= 0) {
                CLDNN_ERROR_MESSAGE(node.id(),
                    "input with iteration axis should not have backedges");
            }
        }
    }
}

layout loop_inst::calc_output_layout(loop_node const & node) {
    // body program should be built here to calculate body input layout
    // from outputs of loop's dependency and calculate loop output layout
    // from the outputs of body program
    if (!node.get_body_program()) {
        node.build_body_program();
    }

    // type checks
    const primitive_id& num_iteration_id = node.get_num_iteration_id();
    if (!node.get_program().get_node(num_iteration_id).is_type<mutable_data>()) {
        CLDNN_ERROR_MESSAGE(node.id(), "num_iteration is not mutable_data");
    }

    if (!check_if_axis_is_set_properly(node)) {
        CLDNN_ERROR_MESSAGE(node.id(), "axis is not set properly");
    }


    const auto& output_primitive_map = node.get_output_primitive_map();

    // assert single output
    // assert(output_primitive_map.size() == 1);
    // // set body network output
    // const auto& body_outputs = node.get_body_program()->get_outputs();
    // for (auto output : body_outputs) {
    //     layout l = output->get_output_layout();
    //     output->set_output_layout(l);
    // }

    // can internal_id and external_id have the same id ?

    // finds internal output
    const auto& output_mapping = output_primitive_map.front();
    const auto& body_outputs = node.get_body_program()->get_outputs();
    const primitive_id& output_internal_id = output_mapping.internal_id;
    auto target = std::find_if(body_outputs.begin(), body_outputs.end(), [&](const cldnn::program_node * output) {
        return output->id() == output_internal_id;
    });
    if (target == body_outputs.end()) {
        CLDNN_ERROR_MESSAGE(node.id(), "output not found");
    }

    // set body output layout
    layout loop_output_layout = (*target)->get_output_layout();
    const int axis_to_iterate_throgh = output_mapping.axis;
    if (axis_to_iterate_throgh != -1) {
        const auto shape = loop_output_layout.size.sizes(loop_output_layout.format);
        const size_t ndim = shape.size();
        const size_t raw_axis = node.convert_to_raw_axis(axis_to_iterate_throgh, ndim);
        loop_output_layout.size.raw[raw_axis] = node.get_max_iteration();
    }
    return loop_output_layout;
}

std::string loop_inst::to_string(const loop_node & node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    // TODO(cldnn loop): loop_inst::to_string
    //   - [x] inputs
    //   - [x] body: body input
    //   - [x] trip_count_id
    //   - [x] initial_execution_id
    //   - [x] current_iteration_id if not empty
    //   - [x] condition_id if not empty
    //   - [] primitive_map [json_composite{external_id, internal_id, axis, ...}]
    //   - [] back_edges [json_composite{from, to}]
    json_composite loop_info;
    loop_info.add("body input id", desc->body.get_primitive_ids());
    loop_info.add("trip_count_id", desc->trip_count_id);
    loop_info.add("initial_execution_id", desc->initial_execution_id);
    loop_info.add("current_iteration_id", desc->current_iteration_id);
    loop_info.add("condition_id", desc->condition_id);
    // TODO(cldnn loop): Fix json_composite to take std::vector<json_composite>()
    // loop_info.add("primitive_map", std::vector<json_composite>());
    // loop_info.add("back_edges", std::vector<json_composite>());


    std::stringstream primitive_description;
    node_info->add("loop info", loop_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

static void validate_primitive_map(loop_node const & node) {
    const auto outer_inputs = node.get_dependencies_ids();
    const auto& input_primitive_map = node.get_input_primitive_map();
    const auto& output_primitive_map = node.get_output_primitive_map();

    // check all loop inputs have their own primitive_map
    for (const auto& id : outer_inputs) {
        if (id == node.get_trip_count_id() ||
            id == node.get_initial_execution_id() ||
            id == node.get_num_iteration_id()) {
            continue;
        }
        const auto results = node.find_primitive_mappings(id, input_primitive_map);
        if (results.size() == 0) {
            std::string msg = "outer input '" + id + "' does not have primitive map";
            CLDNN_ERROR_MESSAGE(node.id(), msg.c_str());
        }
    }

    // check all primitive_mappings have their corresponding external id
    for (const auto& pm : input_primitive_map) {
        auto found = std::find(outer_inputs.begin(), outer_inputs.end(), pm.external_id);
        if (found == outer_inputs.end()) {
            std::string msg = "external id '" + pm.external_id + "' in primitive map cannot be found loop inputs";
            CLDNN_ERROR_MESSAGE(node.id(), msg.c_str());
        }
    }

    const std::list<program_node*>& body_inputs = node.get_body_program()->get_inputs();
    const std::vector<program_node*>& body_outputs = node.get_body_program()->get_outputs();

    // check all primitive_mappings have their corresponding interal id
    for (const auto& pm : input_primitive_map) {
        auto found = std::find_if(body_inputs.begin(), body_inputs.end(), [&pm](const program_node* body_input) {
            return body_input->id() == pm.internal_id;
        });
        if (found == body_inputs.end()) {
            std::string msg = "internal id '" + pm.internal_id + "' in primitive map cannot be found body inputs";
            CLDNN_ERROR_MESSAGE(node.id(), msg.c_str());
        }
    }
    for (const auto& pm : output_primitive_map) {
        auto found = std::find_if(body_outputs.begin(), body_outputs.end(), [&pm](const program_node* body_output) {
            return body_output->id() == pm.internal_id;
        });
        if (found == body_outputs.end()) {
            std::string msg = "internal id '" + pm.internal_id + "' in primitive map cannot be found body outputs";
            CLDNN_ERROR_MESSAGE(node.id(), msg.c_str());
        }
    }
}

loop_inst::typed_primitive_inst(network_impl & network, loop_node const & node)
    : parent(network, node),
      body_network(node.get_program()
        .get_engine()
        .allocate_network(*node.get_body_program(),
                          network.get_stream_id(),
                          false)) {
    // TODO(cldnn loop): move validation code in calc_output_layout to here
    if (!check_if_axis_is_set_properly(node))
        CLDNN_ERROR_MESSAGE(node.id(), "axis is not set properly");

    validate_backedges(node);
    validate_primitive_map(node);
}

}  // namespace cldnn
