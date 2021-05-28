// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "loop_inst.h"

#include "error_handler.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "api/data.hpp"
#include "api/mutable_data.hpp"
#include <string>
#include <exception>
#include <algorithm>

namespace cldnn {
primitive_type_id loop::type_id() {
    static primitive_type_base<loop> instance;
    return &instance;
}

static bool check_if_axis_is_set_properly(loop_node const & node) {
    const auto& input_primitive_maps = node.get_input_primitive_maps();

    std::vector<std::reference_wrapper<const loop::io_primitive_map>> input_with_axis_iteration;
    for (const auto& input : input_primitive_maps) {
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
        const size_t iteration_axis = node.convert_to_raw_axis(pm.get().axis, static_cast<int32_t>(shape.size()));
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
        const loop::io_primitive_map& input = input_ref.get();
        auto dep = std::find_if(dependencies.begin(), dependencies.end(),
            [&input](const cldnn::program_node *dep) { return input.external_id == dep->id(); });

        // if corresponding external id is not found
        if (dep == dependencies.end()) {
            return false;
        }
    }
    return true;
}

static void validate_backedges(loop_node const & node) {
    const auto& back_edges = node.get_back_edges();
    const auto& input_primitive_maps = node.get_input_primitive_maps();

    // check input with iteration axis has backedge
    for (const auto& back_edge : back_edges) {
        for (const auto& mapping : input_primitive_maps) {
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


    // finds internal output
    const auto& output_primitive_maps = node.get_output_primitive_maps();
    const auto& output_mapping = output_primitive_maps.front();
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
    const int64_t axis_to_iterate_throgh = output_mapping.axis;
    if (axis_to_iterate_throgh != -1) {
        const auto shape = loop_output_layout.size.sizes(loop_output_layout.format);
        const size_t ndim = shape.size();
        const size_t raw_axis = node.convert_to_raw_axis(axis_to_iterate_throgh, static_cast<int>(ndim));
        loop_output_layout.size.raw[raw_axis] = static_cast<int32_t>(node.get_max_iteration());
    }
    return loop_output_layout;
}

std::string loop_inst::to_string(const loop_node & node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite loop_info;
    loop_info.add("body input id", desc->body.get_primitive_ids());
    loop_info.add("trip_count_id", desc->trip_count_id);
    loop_info.add("initial_execution_id", desc->initial_execution_id);
    loop_info.add("current_iteration_id", desc->current_iteration_id);
    loop_info.add("condition_id", desc->condition_id);

    std::stringstream primitive_description;
    node_info->add("loop info", loop_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

static void validate_mappings(loop_node const & node) {
    const auto outer_inputs = node.get_dependencies_ids();
    const auto& input_primitive_maps = node.get_input_primitive_maps();
    const auto& output_primitive_maps = node.get_output_primitive_maps();

    // check all loop inputs have their own primitive_map
    for (const auto& id : outer_inputs) {
        if (id == node.get_trip_count_id() ||
            id == node.get_initial_execution_id() ||
            id == node.get_num_iteration_id()) {
            continue;
        }
        const auto results = node.find_io_primitive_maps(id, true);
        if (results.size() == 0) {
            std::string msg = "outer input '" + id + "' does not have primitive map";
            CLDNN_ERROR_MESSAGE(node.id(), msg.c_str());
        }
    }

    // check all io_primitive_maps have their corresponding external id
    for (const auto& pm : input_primitive_maps) {
        auto found = std::find(outer_inputs.begin(), outer_inputs.end(), pm.external_id);
        if (found == outer_inputs.end()) {
            std::string msg = "external id '" + pm.external_id + "' in primitive map cannot be found loop inputs";
            CLDNN_ERROR_MESSAGE(node.id(), msg.c_str());
        }
    }

    const auto& nodes = node.get_body_program()->get_processing_order();

    // check all io_primitive_maps have their corresponding interal id
    for (const auto& pm : input_primitive_maps) {
        auto found = std::find_if(nodes.begin(), nodes.end(), [&pm](const program_node* body_input) {
            return body_input->id() == pm.internal_id;
        });
        if (found == nodes.end()) {
            std::string msg = "internal id '" + pm.internal_id + "' in primitive map cannot be found loop body";
            CLDNN_ERROR_MESSAGE(node.id(), msg.c_str());
        }
    }
    for (const auto& pm : output_primitive_maps) {
        auto found = std::find_if(nodes.begin(), nodes.end(), [&pm](const program_node* body_output) {
            return body_output->id() == pm.internal_id;
        });
        if (found == nodes.end()) {
            std::string msg = "internal id '" + pm.internal_id + "' in primitive map cannot be found body body";
            CLDNN_ERROR_MESSAGE(node.id(), msg.c_str());
        }
    }
}

void loop_inst::preprocess_output_memory() {
    auto& engine = _network.get_engine();
    const auto& output_primitive_maps = node.get_output_primitive_maps();
    concatenated_output_mem_mappings.reserve(output_primitive_maps.size());
    for (size_t i = 0; i < output_primitive_maps.size(); ++i) {
        const auto& output_mapping = output_primitive_maps.at(i);
        const primitive_id& external_id = output_mapping.external_id;
        const primitive_id& internal_id = output_mapping.internal_id;
        if (output_mapping.axis < 0) {
            memory_impl::ptr memory = get_external_memory(external_id);
            body_network->get_primitive(internal_id)->set_output_memory(*memory);
        } else {
            memory_impl::ptr to_mem = get_external_memory(external_id);
            auto output_prim = body_network->get_primitive(internal_id);
            layout sliced_layout = output_prim->output_memory().get_layout();

            const int64_t max_iteration = node.get_max_iteration();
            std::vector<memory_impl::ptr> sliced_mems;
            sliced_mems.reserve(max_iteration);
            for (int j=0; j < max_iteration; ++j) {
                memory_impl::ptr sliced_mem = engine.allocate_memory(sliced_layout, 0);
                sliced_mems.push_back(sliced_mem);
            }

            const int64_t num_elements_batch = concatenated_memory_mapping::get_batch_size(
                sliced_layout, output_mapping.axis);
            const int64_t num_elements_iteration = sliced_layout.count() / num_elements_batch;
            const int64_t start = output_mapping.start < 0? node.get_max_iteration() - 1: output_mapping.start;
            concatenated_memory_mapping memory_mapping_info(
                output_mapping.axis, to_mem, sliced_mems,
                num_elements_iteration, output_mapping.stride, start);
            memory_mapping_info.concat_data_prim = body_network->get_primitive(internal_id);
            concatenated_output_mem_mappings.push_back(memory_mapping_info);
        }
    }
}

void loop_inst::preprocess_input_memory() {
    auto& engine = _network.get_engine();
    auto& iteration_mem = concatenated_input_mem_mappings;
    for (size_t memory_num = 0; memory_num < inputs_memory_count(); memory_num++) {
        const primitive_id& input_external_id = dependencies().at(memory_num)->id();
        auto input_map_ptrs = node.find_io_primitive_maps(input_external_id, true);
        if (input_map_ptrs.size() == 0) {
            if (input_external_id == node.get_trip_count_id() ||
                input_external_id == node.get_initial_execution_id()) {
                continue;
            }
            CLDNN_ERROR_MESSAGE(id(), "loop primitive_map is incomplete");
        }

        memory_impl& memory = input_memory(memory_num);
        for (size_t i = 0; i < input_map_ptrs.size(); ++i) {
            const auto input_map = input_map_ptrs.at(i);
            bool is_concatenated_input = (input_map->axis >= 0);
            if (is_concatenated_input) {
                layout sliced_layout
                    = body_network->get_primitive(input_map->internal_id)->output_memory().get_layout();
                const int64_t max_iteration = node.get_max_iteration();
                std::vector<memory_impl::ptr> sliced_mems;
                sliced_mems.reserve(max_iteration);
                for (int j=0; j < max_iteration; ++j) {
                    memory_impl::ptr sliced_mem = engine.allocate_memory(sliced_layout, 0);
                    sliced_mems.push_back(sliced_mem);
                }
                const int64_t num_elements_batch = concatenated_memory_mapping::get_batch_size(
                    sliced_layout, input_map->axis);
                const int64_t num_elements_iteration = sliced_layout.count() / num_elements_batch;
                const int64_t start = input_map->start < 0? node.get_max_iteration() - 1: input_map->start;
                concatenated_memory_mapping concatenated_input_mem_mapping_info(
                    input_map->axis, (memory_impl::ptr)&memory, sliced_mems,
                    num_elements_iteration, input_map->stride, start);
                concatenated_input_mem_mapping_info.sliced_data_prim = body_network->get_primitive(input_map->internal_id);
                iteration_mem.push_back(concatenated_input_mem_mapping_info);
            } else {
                if (memory.get_layout().data_type != body_network->get_primitive(input_map->internal_id)->output_memory().get_layout().data_type) {
                    CLDNN_ERROR_MESSAGE(id(), "incompatible datatypes");
                }
                body_network->set_input_data(input_map->internal_id, memory);
            }
        }
    }
}

void loop_inst::preprocess_backedge_memory() {
    const auto& back_edges = node.get_back_edges();
    // checking if memory is a destination of a backedge
    for (const auto& back_edge : back_edges) {
        //find corresponding input of the backedge
        const auto input_map_ptrs = node.find_io_primitive_maps(back_edge.to, false);
        assert(input_map_ptrs.size() == 1);
        const auto& input_map = input_map_ptrs.front();
        auto backedged_sliced_output_mems = get_sliced_mem(back_edge.from);
        const auto backedge_to_prim = body_network->get_primitive(back_edge.to);
        const auto backedge_from_prim = body_network->get_primitive(back_edge.from);
        memory_impl::ptr initial_mem = get_external_memory(input_map->external_id);
        if (backedged_sliced_output_mems.empty()) {
            // backedge output which does not need concatenation
            // input memory = output memory = loop output memory
            const auto output_mapping = node.find_io_primitive_maps(back_edge.from, false);
            memory_impl::ptr backedge_mem;
            if (output_mapping.empty()) {
                // from and to primitives in backedge are connected directly
                if (backedge_to_prim == backedge_from_prim->dependencies().front()) {
                    backedge_memory_mappings.emplace_back(
                        backedge_from_prim, backedge_to_prim, initial_mem);
                    continue;
                } else {
                    auto output_prim = body_network->get_primitive(back_edge.from);
                    layout output_layout = output_prim->output_memory().get_layout();
                    backedge_mem = body_network->get_engine().allocate_memory(output_layout, 0);
                }
            } else {
                backedge_mem = get_external_memory(output_mapping.front()->external_id);
            }
            body_network->set_input_data(back_edge.to, *backedge_mem);
            body_network->set_output_memory(back_edge.from, *backedge_mem);
            backedge_memory_mappings.emplace_back(
                backedge_from_prim, backedge_to_prim, backedge_mem, initial_mem);
        } else {
            // backedge output which needs concatenation
            backedge_memory_mappings.emplace_back(
                backedge_from_prim, backedge_to_prim, backedged_sliced_output_mems, initial_mem);
        }
    }
}

std::vector<memory_impl::ptr> loop_inst::get_sliced_mem(const primitive_id& internal_id) const {
        for (const auto& mem_mapping : concatenated_input_mem_mappings) {
            if (mem_mapping.sliced_data_prim->id() == internal_id) {
                return mem_mapping.sliced_mems;
            }
        }
        for (const auto& mem_mapping : concatenated_output_mem_mappings) {
            if (mem_mapping.concat_data_prim->id() == internal_id) {
                return mem_mapping.sliced_mems;
            }
        }
        return {}; // not found
    }

memory_impl::ptr loop_inst::get_external_memory(const primitive_id& external_id) const {
    const auto outputPrim = _network.get_primitive(external_id);
    memory_impl& memory = outputPrim->output_memory();
    return (memory_impl::ptr) &memory;
}

loop_inst::typed_primitive_inst(network_impl & network, loop_node const & node)
    : parent(network, node),
      preproc_memories_done(false),
      body_network(node.get_program()
        .get_engine()
        .allocate_network(*node.get_body_program(),
                          network.get_stream_id(),
                          false)) {
    if (!check_if_axis_is_set_properly(node))
        CLDNN_ERROR_MESSAGE(node.id(), "axis is not set properly");

    validate_backedges(node);
    validate_mappings(node);
}

}  // namespace cldnn
