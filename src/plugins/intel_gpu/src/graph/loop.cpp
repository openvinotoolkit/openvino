// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "loop_inst.h"

#include "json_object.h"
#include "primitive_type_base.h"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/graph/serialization/loop_serializer.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include <string>
#include <exception>
#include <algorithm>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(loop)

static bool check_if_axis_is_set_properly(loop_node const & node) {
    const auto& input_primitive_maps = node.get_input_primitive_maps();

    std::vector<std::reference_wrapper<const loop::io_primitive_map>> input_with_axis_iteration;
    for (const auto& input : input_primitive_maps) {
        if (input.axis >= 0) {
            input_with_axis_iteration.push_back(std::cref(input));
        }
    }

    // check all iteration axis has the same size
    const std::vector<std::pair<program_node*, int32_t>>& dependencies = node.get_dependencies();
    int32_t iteration_size = -1;
    for (const auto& pm : input_with_axis_iteration) {
        auto found = std::find_if(dependencies.begin(), dependencies.end(),
            [&pm](const std::pair<program_node*, int32_t>& dep){ return dep.first->id() == pm.get().external_id; });
        assert(found != dependencies.end());
        const layout input_layout = (*found).first->get_output_layout();
        const auto shape = input_layout.get_tensor().sizes(input_layout.format);
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
            [&input](const std::pair<program_node*, int>& dep) { return input.external_id == dep.first->id(); });

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

layout loop_inst::calc_output_layout(loop_node const & node, kernel_impl_params const& impl_param) {
    // body program should be built here to calculate body input layout
    // from outputs of loop's dependency and calculate loop output layout
    // from the outputs of body program
    if (!node.get_body_program()) {
        const_cast<loop_node&>(node).build_body_program();
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
        CLDNN_ERROR_MESSAGE(impl_param.desc->id, "output not found");
    }

    // set body output layout
    layout loop_output_layout = (*target)->get_output_layout();
    const int64_t axis_to_iterate_throgh = output_mapping.axis;
    if (axis_to_iterate_throgh != -1) {
        const size_t ndim = loop_output_layout.get_rank();
        auto shape = loop_output_layout.get_dims();
        shape[axis_to_iterate_throgh] = static_cast<int32_t>(node.get_max_iteration());
        loop_output_layout.set_tensor(tensor(format::get_default_format(ndim), shape));
    }
    return loop_output_layout;
}

std::string loop_inst::to_string(const loop_node & node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite loop_info;
    loop_info.add("body input id", desc->body.get_primitives_ids());
    loop_info.add("trip_count_id", desc->trip_count_id);
    loop_info.add("initial_execution_id", desc->initial_execution_id);
    loop_info.add("current_iteration_id", desc->current_iteration_id);
    loop_info.add("condition_id", desc->condition_id);

    std::stringstream primitive_description;
    node_info->add("loop info", loop_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

static std::vector<const loop::io_primitive_map*> find_io_primitive_maps(
                                                    const std::vector<loop::io_primitive_map>& input_primitive_maps,
                                                    const std::vector<loop::io_primitive_map>& output_primitive_maps,
                                                    const primitive_id& prim_id,
                                                    bool is_external) {
    std::vector<const loop::io_primitive_map*> ret;
    if (is_external) {
        for (const auto& it : input_primitive_maps) {
            if (it.external_id == prim_id) {
                ret.push_back(&it);
            }
        }
        for (const auto& it : output_primitive_maps) {
            if (it.external_id == prim_id) {
                ret.push_back(&it);
            }
        }
    } else {
        for (const auto& it : input_primitive_maps) {
            if (it.internal_id == prim_id) {
                ret.push_back(&it);
            }
        }
        for (const auto& it : output_primitive_maps) {
            if (it.internal_id == prim_id) {
                ret.push_back(&it);
            }
        }
    }
    return ret;
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
        const auto results = find_io_primitive_maps(node.get_input_primitive_maps(),
                                                    node.get_output_primitive_maps(), id, true);
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

void loop_inst::update_mapped_memory() {
    if (!preproc_memories_done) {
        return;
    }
    // update output memory
    for (size_t i = 0; i < _output_primitive_maps.size(); ++i) {
        const auto& output_mapping = _output_primitive_maps.at(i);
        const primitive_id& external_id = output_mapping.external_id;
        const primitive_id& internal_id = output_mapping.internal_id;
        memory::ptr to_mem = get_external_memory(external_id);
        if (output_mapping.axis < 0) {
            body_network->get_primitive(internal_id)->set_output_memory(to_mem);
        } else {
            for (auto& mem_mapping : concatenated_output_mem_mappings) {
                if (mem_mapping.concat_data_prim->id() == internal_id) {
                    mem_mapping.concatenated_mem = to_mem;
                    break;
                }
            }
        }
    }
    // update input memory
    for (size_t memory_num = 0; memory_num < inputs_memory_count(); memory_num++) {
        const primitive_id& input_external_id = dependencies().at(memory_num).first->id();
        auto input_map_ptrs = find_io_primitive_maps(_input_primitive_maps,
                                                     _output_primitive_maps, input_external_id, true);
        if (input_map_ptrs.empty()) {
            if (input_external_id == _trip_count_id ||
                input_external_id == _initial_execution_id) {
                continue;
            }
        }

        auto memory = input_memory_ptr(memory_num);
        for (size_t i = 0; i < input_map_ptrs.size(); ++i) {
            const auto input_map = input_map_ptrs.at(i);
            bool is_concatenated_input = (input_map->axis >= 0);
            if (is_concatenated_input) {
                for (auto& mem_mapping : concatenated_input_mem_mappings) {
                    if (mem_mapping.sliced_data_prim->id() == input_map->internal_id) {
                        mem_mapping.concatenated_mem = memory;
                        break;
                    }
                }
            } else {
                body_network->set_input_data(input_map->internal_id, memory);
            }
        }
    }
    //update backedges memory
    // checking if memory is a destination of a backedge
    for (const auto& back_edge : _back_edges) {
        //find corresponding input of the backedge
        const auto input_map_ptrs = find_io_primitive_maps(_input_primitive_maps,
                                                           _output_primitive_maps, back_edge.to, false);
        assert(input_map_ptrs.size() == 1);
        const auto& input_map = input_map_ptrs.front();
        auto backedged_sliced_output_mems = get_sliced_mem(back_edge.from);
        const auto backedge_to_prim = body_network->get_primitive(back_edge.to);
        const auto backedge_from_prim = body_network->get_primitive(back_edge.from);
        memory::ptr initial_mem = get_external_memory(input_map->external_id);

        for (auto& backedge_mapping : backedge_memory_mappings) {
            if (backedge_mapping.from_primitive->id() == backedge_from_prim->id() &&
                backedge_mapping.to_primitive->id() == backedge_to_prim->id()) {
                if (backedged_sliced_output_mems.empty()) {
                    // backedge output which does not need concatenation
                    const auto output_mapping = find_io_primitive_maps(_input_primitive_maps,
                                                                       _output_primitive_maps, back_edge.from, false);
                    memory::ptr backedge_mem;
                    if (output_mapping.empty()) {
                        // from and to primitives in backedge are connected directly
                        if (backedge_to_prim == backedge_from_prim->dependencies().front().first) {
                            backedge_mapping.initial_mem = initial_mem;
                            continue;
                        } else {
                            // generally, shouldn't go this way, but...
                            auto output_prim = body_network->get_primitive(back_edge.from);
                            layout output_layout = output_prim->output_memory().get_layout();
                            backedge_mem = body_network->get_engine().allocate_memory(output_layout, 0);
                        }
                    } else {
                        backedge_mem = get_external_memory(output_mapping.front()->external_id);
                    }
                    body_network->set_input_data(back_edge.to, backedge_mem);
                    body_network->set_output_memory(back_edge.from, backedge_mem);
                    backedge_mapping.from_mems = { backedge_mem };
                    backedge_mapping.initial_mem = initial_mem;
                } else {
                    backedge_mapping.from_mems = backedged_sliced_output_mems;
                    backedge_mapping.initial_mem = initial_mem;
                }
                break;
            }
        }
    }
}

void loop_inst::set_output_memory(memory::ptr mem, bool check, size_t idx) {
    primitive_inst::set_output_memory(mem, check, idx);
    update_mapped_memory();
}

void loop_inst::preprocess_output_memory() {
    auto& engine = _network.get_engine();
    concatenated_output_mem_mappings.reserve(_output_primitive_maps.size());
    for (size_t i = 0; i < _output_primitive_maps.size(); ++i) {
        const auto& output_mapping = _output_primitive_maps.at(i);
        const primitive_id& external_id = output_mapping.external_id;
        const primitive_id& internal_id = output_mapping.internal_id;
        if (output_mapping.axis < 0) {
            memory::ptr memory = get_external_memory(external_id);
            body_network->get_primitive(internal_id)->set_output_memory(memory);
        } else {
            memory::ptr to_mem = get_external_memory(external_id);
            auto output_prim = body_network->get_primitive(internal_id);
            layout sliced_layout = output_prim->output_memory().get_layout();

            const int64_t max_iteration = _max_iteration;
            std::vector<memory::ptr> sliced_mems;
            sliced_mems.reserve(max_iteration);
            for (int j=0; j < max_iteration; ++j) {
                memory::ptr sliced_mem = engine.allocate_memory(sliced_layout, 0);
                sliced_mems.push_back(sliced_mem);
            }

            const int64_t num_elements_batch = concatenated_memory_mapping::get_batch_size(
                sliced_layout, output_mapping.axis);
            const int64_t num_elements_iteration = sliced_layout.count() / num_elements_batch;
            const int64_t start = output_mapping.start < 0? _max_iteration - 1: output_mapping.start;
            concatenated_memory_mapping memory_mapping_info(
                output_mapping.axis, to_mem, sliced_mems, _network.get_stream(),
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
        const primitive_id& input_external_id = dependencies().at(memory_num).first->id();
        auto input_map_ptrs = find_io_primitive_maps(_input_primitive_maps,
                                                     _output_primitive_maps, input_external_id, true);
        if (input_map_ptrs.size() == 0) {
            if (input_external_id == _trip_count_id ||
                input_external_id == _initial_execution_id) {
                continue;
            }
            CLDNN_ERROR_MESSAGE(id(), "loop primitive_map is incomplete");
        }

        auto memory = input_memory_ptr(memory_num);
        for (size_t i = 0; i < input_map_ptrs.size(); ++i) {
            const auto input_map = input_map_ptrs.at(i);
            bool is_concatenated_input = (input_map->axis >= 0);
            if (is_concatenated_input) {
                layout sliced_layout
                    = body_network->get_primitive(input_map->internal_id)->output_memory().get_layout();
                const int64_t max_iteration = _max_iteration;
                std::vector<memory::ptr> sliced_mems;
                sliced_mems.reserve(max_iteration);
                for (int j=0; j < max_iteration; ++j) {
                    memory::ptr sliced_mem = engine.allocate_memory(sliced_layout, 0);
                    sliced_mems.push_back(sliced_mem);
                }
                const int64_t num_elements_batch = concatenated_memory_mapping::get_batch_size(
                    sliced_layout, input_map->axis);
                const int64_t num_elements_iteration = sliced_layout.count() / num_elements_batch;
                const int64_t start = input_map->start < 0? _max_iteration - 1: input_map->start;
                concatenated_memory_mapping concatenated_input_mem_mapping_info(
                    input_map->axis, memory, sliced_mems, _network.get_stream(),
                    num_elements_iteration, input_map->stride, start);
                concatenated_input_mem_mapping_info.sliced_data_prim = body_network->get_primitive(input_map->internal_id);
                iteration_mem.push_back(concatenated_input_mem_mapping_info);
            } else {
                if (memory->get_layout().data_type != body_network->get_primitive(input_map->internal_id)->output_memory().get_layout().data_type) {
                    CLDNN_ERROR_MESSAGE(id(), "incompatible datatypes");
                }
                body_network->set_input_data(input_map->internal_id, memory);
            }
        }
    }
}

void loop_inst::preprocess_backedge_memory() {
    // checking if memory is a destination of a backedge
    for (const auto& back_edge : _back_edges) {
        //find corresponding input of the backedge
        const auto input_map_ptrs = find_io_primitive_maps(_input_primitive_maps,
                                                           _output_primitive_maps, back_edge.to, false);
        const auto backedge_to_prim = body_network->get_primitive(back_edge.to);
        const auto backedge_from_prim = body_network->get_primitive(back_edge.from);

        memory::ptr initial_mem;
        if (back_edge.to == _current_iteration_id) {
            const layout current_iteration_layout = backedge_to_prim->output_memory().get_layout();
            initial_mem = get_network().get_engine().allocate_memory(current_iteration_layout);
            auto& stream = get_network().get_stream();
            loop_node::write_scalar_value(initial_mem, stream, 0);
            current_iteratoin_backedge_mapping_idx = backedge_memory_mappings.size();
        } else {
            if (input_map_ptrs.empty()) {
                CLDNN_ERROR_MESSAGE(id(), "no input_mapping for backedged input");
            }
            initial_mem = get_external_memory(input_map_ptrs.front()->external_id);
        }

        auto backedged_sliced_output_mems = get_sliced_mem(back_edge.from);
        if (backedged_sliced_output_mems.empty()) {
            // backedge output which does not need concatenation
            const auto output_mapping = find_io_primitive_maps(_input_primitive_maps,
                                                               _output_primitive_maps, back_edge.from, false);
            memory::ptr backedge_mem;
            if (output_mapping.empty()) {
                // from and to primitives in backedge are connected directly
                if (backedge_to_prim == backedge_from_prim->dependencies().front().first) {
                    backedge_memory_mappings.emplace_back(
                        backedge_from_prim, backedge_to_prim, initial_mem, body_network->get_stream());
                    continue;
                } else {
                    auto output_prim = body_network->get_primitive(back_edge.from);
                    layout output_layout = output_prim->output_memory().get_layout();
                    backedge_mem = body_network->get_engine().allocate_memory(output_layout, 0);
                }
            } else {
                backedge_mem = get_external_memory(output_mapping.front()->external_id);
            }
            body_network->set_input_data(back_edge.to, backedge_mem);
            body_network->set_output_memory(back_edge.from, backedge_mem);
            backedge_memory_mappings.emplace_back(
                backedge_from_prim, backedge_to_prim, backedge_mem, initial_mem, body_network->get_stream());
        } else {
            // backedge output which needs concatenation
            backedge_memory_mappings.emplace_back(
                backedge_from_prim, backedge_to_prim, backedged_sliced_output_mems, initial_mem, body_network->get_stream());
        }
    }
}

std::vector<memory::ptr> loop_inst::get_sliced_mem(const primitive_id& internal_id) const {
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

memory::ptr loop_inst::get_external_memory(const primitive_id& external_id) const {
    const auto outputPrim = _network.get_primitive(external_id);
    return outputPrim->output_memory_ptr();
}

loop_inst::typed_primitive_inst(network & network, loop_node const & node)
    : parent(network, node),
      preproc_memories_done(false),
      body_network(network::allocate_network(network.get_stream_ptr(),
                                                  node.get_body_program(),
                                                  false,
                                                  network.is_primary_stream())) {
    if (!check_if_axis_is_set_properly(node))
        CLDNN_ERROR_MESSAGE(node.id(), "axis is not set properly");

    validate_backedges(node);
    validate_mappings(node);

    _input_primitive_maps = node.get_input_primitive_maps();
    _output_primitive_maps = node.get_output_primitive_maps();
    _back_edges = node.get_back_edges();
    _trip_count_id = node.get_trip_count_id();
    _initial_execution_id = node.get_initial_execution_id();
    _current_iteration_id = node.get_current_iteration_id();
    _condition_id = node.get_condition_id();
    _num_iteration_id = node.get_num_iteration_id();
    _max_iteration = node.get_max_iteration();
}

void loop_inst::save(BinaryOutputBuffer& ob) const {
    parent::save(ob);
    ob << _input_primitive_maps;
    ob << _output_primitive_maps;
    ob << _back_edges;
    ob << _trip_count_id;
    ob << _initial_execution_id;
    ob << _current_iteration_id;
    ob << _condition_id;
    ob << _num_iteration_id;
    ob << _max_iteration;
    body_network->save(ob);
}

void loop_inst::load(BinaryInputBuffer& ib) {
    parent::load(ib);
    preproc_memories_done = false,
    ib >> _input_primitive_maps;
    ib >> _output_primitive_maps;
    ib >> _back_edges;
    ib >> _trip_count_id;
    ib >> _initial_execution_id;
    ib >> _current_iteration_id;
    ib >> _condition_id;
    ib >> _num_iteration_id;
    ib >> _max_iteration;
    body_network = std::make_shared<cldnn::network>(ib, get_network().get_stream_ptr(), get_network().get_engine());
}

}  // namespace cldnn
