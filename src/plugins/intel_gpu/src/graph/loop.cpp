// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "loop_inst.h"

#include "data_inst.h"
#include "mutable_data_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include <string>
#include <exception>
#include <algorithm>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(loop)

std::map<size_t, memory::ptr> loop_node::get_memory_deps() const {
    auto memory_deps = get_const_memory_deps();
    for (auto& i : get_shape_infer_dependencies()) {
        auto& dep = get_dependency(i);
        auto dep_id = dep.id();
        if (memory_deps.count(i) > 0 || i >= get_dependencies().size()) {
            continue;
        }

        memory::ptr mem_ptr = nullptr;
        if (dep.is_type<data>()) {
            mem_ptr = dep.as<data>().get_attached_memory_ptr();
        } else if (dep.is_type<mutable_data>()) {
            mem_ptr = dep.as<mutable_data>().get_attached_memory_ptr();
        }

        if (mem_ptr) {
            memory_deps.insert({i, mem_ptr});
        }
    }
    return memory_deps;
}

static size_t convert_to_raw_axis(size_t axis, size_t ndim) {
    // convert between bfyx, bfzyx, bfzyxw and tensor.size.raw
    if (axis >= ndim) {
        throw std::runtime_error("axis should be less than ndim");
    }

    if (axis < 2) {
        return axis;
    }
    return (ndim - 1) - (axis - 2);
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
    const std::vector<std::pair<program_node*, int32_t>>& dependencies = node.get_dependencies();
    int32_t iteration_size = -1;
    for (const auto& pm : input_with_axis_iteration) {
        auto found = std::find_if(dependencies.begin(), dependencies.end(),
            [&pm](const std::pair<program_node*, int32_t>& dep){ return dep.first->id() == pm.get().external_id.pid; });
        assert(found != dependencies.end());
        const layout input_layout = (*found).first->get_output_layout();
        const auto shape = input_layout.get_tensor().sizes(input_layout.format);
        const size_t iteration_axis = convert_to_raw_axis(pm.get().axis, static_cast<int32_t>(shape.size()));
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
            [&input](const std::pair<program_node*, int>& dep) { return input.external_id.pid == dep.first->id(); });

        // if corresponding external id is not found
        if (dep == dependencies.end()) {
            return false;
        }
    }
    return true;
}

layout loop_inst::calc_output_layout(loop_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto prim = impl_param.typed_desc<loop>();

    // finds internal output
    const auto& output_primitive_maps = prim->output_primitive_maps;
    const auto& output_mapping = output_primitive_maps.front();

    const auto& body_program = impl_param.inner_progs.front();
    const auto& body_outputs = body_program->get_outputs();

    const primitive_id& output_internal_id = output_mapping.internal_id.pid;
    auto target = std::find_if(body_outputs.begin(), body_outputs.end(), [&](const cldnn::program_node * output) {
        return output->id() == output_internal_id;
    });
    OPENVINO_ASSERT(target != body_outputs.end(), impl_param.desc->id, "output not found");

    // set body output layout
    layout loop_output_layout = (*target)->get_output_layout();
    const int64_t axis_to_iterate_through = output_mapping.axis;
    if (axis_to_iterate_through != -1) {
        const size_t ndim = loop_output_layout.get_rank();
        auto shape = loop_output_layout.get_dims();
        shape[axis_to_iterate_through] = static_cast<int32_t>(prim->max_num_iterations);
        loop_output_layout.set_tensor(tensor(format::get_default_format(ndim), shape));
    }

    return loop_output_layout;
}

template<typename T>
static std::vector<layout> get_output_layouts(kernel_impl_params const& impl_param, std::vector<T> body_outputs, const int64_t num_iterations = -1) {
    auto prim = impl_param.typed_desc<loop>();
    std::vector<layout> output_layouts;

    const auto& output_primitive_maps = prim->output_primitive_maps;
    for (auto& output_mapping : output_primitive_maps) {
        const primitive_id& output_internal_id = output_mapping.internal_id.pid;
        auto target = std::find_if(body_outputs.begin(), body_outputs.end(), [&](const T output) {
            return output->id() == output_internal_id;
        });
        OPENVINO_ASSERT(target != body_outputs.end(), impl_param.desc->id, "output not found");

        // set body output layout
        layout loop_output_layout = (*target)->get_output_layout();
        const int64_t axis_to_iterate_through = output_mapping.axis;
        if (axis_to_iterate_through != -1) {
            auto shape = loop_output_layout.get_partial_shape();
            shape[axis_to_iterate_through] = static_cast<int32_t>(num_iterations);
            loop_output_layout.set_partial_shape(shape);
        }
        output_layouts.push_back(loop_output_layout);
    }
    return output_layouts;
}

template<typename ShapeType>
std::vector<layout> loop_inst::calc_output_layouts(loop_node const& /*node*/, kernel_impl_params const& impl_param) {
    std::vector<layout> output_layouts;
    auto prim = impl_param.typed_desc<loop>();
    if (impl_param.inner_nets.empty()) {
        OPENVINO_ASSERT(impl_param.inner_progs.size() == 1, "Loop(", prim->id, ") should have only one inner network");
        const auto& body_outputs = impl_param.inner_progs.front()->get_outputs();
        output_layouts = get_output_layouts<program_node*>(impl_param, body_outputs);
    } else {
        auto& memory_deps = impl_param.memory_deps;
        const size_t current_iteration_idx = 0;
        OPENVINO_ASSERT(memory_deps.count(current_iteration_idx) > 0, "The count of memory deps(current_iteration) should not be zero");
        cldnn::mem_lock<int64_t, mem_lock_type::read> current_iterations_lock(memory_deps.at(current_iteration_idx), impl_param.get_stream());
        int64_t current_iteration = static_cast<int64_t>(*current_iterations_lock.data());
        GPU_DEBUG_LOG << "* current_iteration(" << memory_deps.at(current_iteration_idx) << ")  : " << current_iteration << std::endl;

        OPENVINO_ASSERT(impl_param.inner_nets.size() == 1, "Loop(", prim->id, ") should have only one inner program");
        const auto& body_outputs = impl_param.inner_nets.front()->get_outputs();
        output_layouts = get_output_layouts<std::shared_ptr<primitive_inst>>(impl_param, body_outputs, current_iteration);
    }
    return output_layouts;
}

template std::vector<layout> loop_inst::calc_output_layouts<ov::PartialShape>(loop_node const& node, const kernel_impl_params& impl_param);


std::string loop_inst::to_string(const loop_node & node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::vector<primitive_id> body_inputs;
    {
        for (auto& input : desc->body_program->get_inputs()) {
            body_inputs.push_back(input->id());
        }
    }

    json_composite loop_info;
    loop_info.add("body input id", body_inputs);
    loop_info.add("trip_count_id", desc->trip_count_id);
    loop_info.add("first_execution_condition_id", desc->first_execution_condition_id);
    loop_info.add("body_current_iteration_id", desc->body_current_iteration_id);
    loop_info.add("body_execution_condition_id", desc->body_execution_condition_id);

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
            if (it.external_id.pid == prim_id) {
                ret.push_back(&it);
            }
        }
        for (const auto& it : output_primitive_maps) {
            if (it.external_id.pid == prim_id) {
                ret.push_back(&it);
            }
        }
    } else {
        for (const auto& it : input_primitive_maps) {
            if (it.internal_id.pid == prim_id) {
                ret.push_back(&it);
            }
        }
        for (const auto& it : output_primitive_maps) {
            if (it.internal_id.pid == prim_id) {
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
            id == node.get_num_iterations_id()) {
            continue;
        }
        const auto results = find_io_primitive_maps(node.get_input_primitive_maps(),
                                                    node.get_output_primitive_maps(), id, true);
        OPENVINO_ASSERT(results.size() > 0, node.id(), " : outer input '", id, "' does not have primitive map");
    }

    // check all io_primitive_maps have their corresponding external id
    for (const auto& pm : input_primitive_maps) {
        auto found = std::find(outer_inputs.begin(), outer_inputs.end(), pm.external_id.pid);
        OPENVINO_ASSERT(found != outer_inputs.end(), node.id(),
                        " : external id '", pm.external_id.pid, "' in primitive map cannot be found loop inputs");
    }

    const auto& nodes = node.get_body_program()->get_processing_order();

    // check all io_primitive_maps have their corresponding interal id
    for (const auto& pm : input_primitive_maps) {
        auto found = std::find_if(nodes.begin(), nodes.end(), [&pm](const program_node* body_input) {
            return body_input->id() == pm.internal_id.pid;
        });
        OPENVINO_ASSERT(found != nodes.end(), node.id(),
                    " : internal id '", pm.internal_id.pid, "' in primitive map cannot be found loop body");
    }
    for (const auto& pm : output_primitive_maps) {
        auto found = std::find_if(nodes.begin(), nodes.end(), [&pm](const program_node* body_output) {
            return body_output->id() == pm.internal_id.pid;
        });
        OPENVINO_ASSERT(found != nodes.end(), node.id(),
                    " : internal id '", pm.internal_id.pid, "' in primitive map cannot be found body body");
    }
}

void loop_inst::update_input_mapped_memory() {
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
                    if (mem_mapping->sliced_data_prim->id() == input_map->internal_id.pid) {
                        mem_mapping->update_concatenated_mem(memory);
                        break;
                    }
                }
            } else {
                body_network->set_input_data(input_map->internal_id.pid, memory);
            }
        }
    }
}

void loop_inst::update_output_mapped_memory() {
    OPENVINO_ASSERT(outputs_allocated(), "output buffer should be allocated");
    for (size_t i = 0; i < _output_primitive_maps.size(); ++i) {
        const auto& output_mapping = _output_primitive_maps.at(i);
        const primitive_id& external_id = output_mapping.external_id.pid;
        const size_t external_mem_idx = output_mapping.external_id.idx;
        const primitive_id& internal_id = output_mapping.internal_id.pid;
        const size_t internal_mem_idx = output_mapping.internal_id.idx;

        memory::ptr to_mem = get_external_memory(external_id, external_mem_idx);
        if (to_mem) {
            if (output_mapping.axis < 0) {
                body_network->get_primitive(internal_id)->set_output_memory(to_mem, true, internal_mem_idx);
            } else {
                for (auto& mem_mapping : concatenated_output_mem_mappings) {
                    if (mem_mapping->sliced_data_prim->id() == internal_id) {
                        mem_mapping->update_concatenated_mem(to_mem);
                        break;
                    }
                }
            }
        }
    }
}

void loop_inst::update_backedge_mapped_memory() {
    // checking if memory is a destination of a backedge
    for (const auto& back_edge : _back_edges) {
        //find corresponding input of the backedge
        const auto input_map_ptrs = find_io_primitive_maps(_input_primitive_maps,
                                                            _output_primitive_maps, back_edge.to, false);
        assert(input_map_ptrs.size() == 1);
        const auto& input_map = input_map_ptrs.front();
        auto backedged_sliced_output = get_sliced_mem(back_edge.from);
        const auto backedge_to_prim = body_network->get_primitive(back_edge.to);
        const auto backedge_from_prim = body_network->get_primitive(back_edge.from);

        memory::ptr initial_mem = get_external_memory(input_map->external_id.pid, input_map->external_id.idx);

        for (auto& backedge_mapping : backedge_memory_mappings) {
            if (backedge_mapping.from_primitive->id() == backedge_from_prim->id() &&
                backedge_mapping.to_primitive->id() == backedge_to_prim->id()) {
                if (backedged_sliced_output == nullptr) {
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
                        auto external_id = output_mapping.front()->external_id;
                        backedge_mem = get_external_memory(external_id.pid, external_id.idx);
                    }
                    body_network->set_input_data(back_edge.to, backedge_mem);
                    body_network->set_output_memory(back_edge.from, backedge_mem);
                    backedge_mapping.from_mem = backedge_mem;
                    backedge_mapping.initial_mem = initial_mem;
                } else {
                    backedge_mapping.concat_mem_mapping = backedged_sliced_output;
                    backedge_mapping.initial_mem = initial_mem;
                }
                break;
            }
        }
    }
}


void loop_inst::update_mapped_memory() {
    if (!preproc_memories_done) {
        return;
    }

    update_output_mapped_memory();
    update_input_mapped_memory();
    update_backedge_mapped_memory();
}

event::ptr loop_inst::set_output_memory(memory::ptr mem, bool check, size_t idx) {
    auto ev = primitive_inst::set_output_memory(mem, check, idx);
    update_mapped_memory();
    return ev;
}

loop_inst::concatenated_memory_mapping::ptr loop_inst::create_concat_memory_map(const input_info& internal_id,
                                        const cldnn::loop::io_primitive_map& io_prim_map,
                                        memory::ptr mem_ptr,
                                        const int64_t trip_count) {
    auto& engine = body_network->get_engine();
    auto& stream = body_network->get_stream();
    auto prim = body_network->get_primitive(internal_id.pid);
    const int64_t start = io_prim_map.start < 0? trip_count - 1: io_prim_map.start;

    std::vector<memory::ptr> sliced_mems;
    int64_t num_elements_iteration = 0;

    // if memory is nullptr, that means memory is not allocated yet because current network is dynamic shape model.
    // In dynamic model, we can't calculate num_element_iteration, start, and sliced_layout.
    // will recalculate that parameters in backedge preprocessing map after first execution.
    if (mem_ptr != nullptr) {
        layout sliced_layout = prim->output_memory(internal_id.idx).get_layout();

        // When trip_count is -1, allocate first sliced_mem and allocate sliced memory if additional sliced mem is required
        if (trip_count < 0) {
            memory::ptr sliced_mem = engine.allocate_memory(sliced_layout, 0);
            sliced_mems.push_back(sliced_mem);
        } else {
            sliced_mems.reserve(trip_count);
            for (int j=0; j < trip_count; ++j) {
                memory::ptr sliced_mem = engine.allocate_memory(sliced_layout, 0);
                sliced_mems.push_back(sliced_mem);
            }
        }

        const int64_t num_elements_batch = concatenated_memory_mapping::get_batch_size(
            sliced_layout, io_prim_map.axis);
        num_elements_iteration = sliced_layout.count() / num_elements_batch;
    }

    auto concat_memory_mapping = std::make_shared<concatenated_memory_mapping>(
                                                    io_prim_map.axis, mem_ptr, sliced_mems, stream,
                                                    engine, num_elements_iteration, io_prim_map.stride, start);
    concat_memory_mapping->sliced_data_prim = body_network->get_primitive(internal_id.pid);
    return concat_memory_mapping;
}

void loop_inst::preprocess_output_memory(const int64_t trip_count) {
    if (concatenated_output_mem_mappings.empty())
        concatenated_output_mem_mappings.reserve(_output_primitive_maps.size());
    for (size_t i = 0; i < _output_primitive_maps.size(); ++i) {
        const auto& output_mapping = _output_primitive_maps.at(i);
        const auto& external_id = output_mapping.external_id;
        const auto& internal_id = output_mapping.internal_id;
        GPU_DEBUG_LOG << i << ") output mapping - external " << external_id.to_string() << std::endl;
        GPU_DEBUG_LOG << i << ") output mapping - internal " << internal_id.to_string() << std::endl;

        memory::ptr memory = get_external_memory(external_id.pid, external_id.idx);
        if (output_mapping.axis < 0) {
            // In dynamic model, Don't get output memory of loop node because body network's output layouts are not calculated
            if (memory != nullptr) {
                body_network->get_primitive(internal_id.pid)->set_output_memory(memory, true, internal_id.idx);
            }
        } else {
            auto iter = std::find_if(concatenated_output_mem_mappings.begin(), concatenated_output_mem_mappings.end(),
                [&](loop_inst::concatenated_memory_mapping::ptr concat_mem_map) -> bool {
                    return concat_mem_map->sliced_data_prim->id() == internal_id.pid;
                });
            if (iter == concatenated_output_mem_mappings.end()) {
                auto memory_mapping_info = create_concat_memory_map(internal_id, output_mapping, memory, trip_count);
                memory_mapping_info->concat_data_prim = get_network().get_primitive(external_id.pid);
                memory_mapping_info->concat_data_id = external_id;
                concatenated_output_mem_mappings.push_back(memory_mapping_info);
                GPU_DEBUG_LOG << i << ") generate concat output memory mapping: " << memory_mapping_info->to_string() << std::endl;
            }
            GPU_DEBUG_IF(iter != concatenated_output_mem_mappings.end()) {
                GPU_DEBUG_LOG << i << ") memory_mapping_info is already existed : " << (*iter)->to_string() << std::endl;
            }
        }
    }
}

void loop_inst::preprocess_input_memory(const int64_t trip_count) {
    for (size_t memory_num = 0; memory_num < inputs_memory_count(); memory_num++) {
        const primitive_id& input_external_id = dependencies().at(memory_num).first->id();
        auto input_map_ptrs = find_io_primitive_maps(_input_primitive_maps,
                                                    _output_primitive_maps, input_external_id, true);
        if (input_map_ptrs.size() == 0) {
            OPENVINO_ASSERT((input_external_id == _trip_count_id
                                || input_external_id == _num_iterations_id
                                || input_external_id == _initial_execution_id),
                                id(), "loop primitive_map is incomplete "
                                "input_external_id(", input_external_id, ") != _trip_count_id(", _trip_count_id, ")",
                                "input_external_id(", input_external_id, ") != _num_iterations_id(", _num_iterations_id, ")",
                                " && input_external_id(", input_external_id, ") != _initial_execution_id(", _initial_execution_id, ")");
            continue;
        }

        auto memory = input_memory_ptr(memory_num);
        for (size_t i = 0; i < input_map_ptrs.size(); ++i) {
            const auto input_map = input_map_ptrs.at(i);
            const auto& external_id = input_map->external_id;
            const auto& internal_id = input_map->internal_id;
            GPU_DEBUG_LOG << i << ") input mapping - external " << external_id.to_string() << std::endl;
            GPU_DEBUG_LOG << i << ") input mapping - internal " << internal_id.to_string() << std::endl;

            if (input_map->axis >= 0) {
                OPENVINO_ASSERT(trip_count > 0, "In preprocessing concat input mapping, trip_count should be positive");
                OPENVINO_ASSERT(memory != nullptr, "In preprocessing concat input mapping, concat memory should be allocated");
                auto memory_mapping_info = create_concat_memory_map(internal_id, *input_map, memory, trip_count);
                concatenated_input_mem_mappings.push_back(memory_mapping_info);
                GPU_DEBUG_LOG << i << ") generate concat input memory mapping: " << memory_mapping_info->to_string() << std::endl;
            } else {
                auto input_inst = body_network->get_primitive(internal_id.pid);
                if (memory->get_layout() != input_inst->get_output_layout()) {
                    input_inst->set_output_layout(memory->get_layout());
                    GPU_DEBUG_LOG << input_inst->id() << " is changed memory because layout is changed from "
                                        << input_inst->get_output_layout().to_short_string()
                                        << " to " << memory->get_layout().to_short_string() << std::endl;
                }
                body_network->set_input_data(internal_id.pid, memory);
            }
        }
    }
}

void loop_inst::preprocess_backedge_memory() {
    // checking if memory is a destination of a backedge
    for (size_t idx = 0; idx < _back_edges.size(); idx++) {
        const auto& back_edge = _back_edges[idx];
        //find corresponding input of the backedge
        const auto input_map_ptrs = find_io_primitive_maps(_input_primitive_maps,
                                                            _output_primitive_maps, back_edge.to, false);
        const auto backedge_to_prim = body_network->get_primitive(back_edge.to);
        const auto backedge_from_prim = body_network->get_primitive(back_edge.from);

        memory::ptr initial_mem;
        OPENVINO_ASSERT(!input_map_ptrs.empty(), id(), " has no input_mapping for backedged input");
        auto& external_id = input_map_ptrs.front()->external_id;
        initial_mem = get_external_memory(external_id.pid, external_id.idx);

        GPU_DEBUG_LOG << idx << ") back_edge mapping - back_edge.from " << back_edge.from << std::endl;
        GPU_DEBUG_LOG << idx << ") back_edge mapping - back_edge.to   " << back_edge.to << std::endl;

        auto backedged_sliced_output = get_sliced_mem(back_edge.from);
        const auto output_mapping = find_io_primitive_maps(_input_primitive_maps,
                                                            _output_primitive_maps, back_edge.from, false);
        if (backedged_sliced_output != nullptr) {
            // CONCAT_OUTPUT mode, backedge output which needs concatenation
            backedge_memory_mappings.emplace_back(
                backedge_from_prim, backedge_to_prim, backedged_sliced_output, initial_mem, body_network->get_stream());
            GPU_DEBUG_LOG << idx << ") add back_edge mapping with CONCAT_OUTPUT type, backedged_sliced_output("
                            << backedged_sliced_output << "), initial_mem(" << initial_mem << ")" << std::endl;
        } else if (output_mapping.empty() && backedge_to_prim == backedge_from_prim->dependencies().front().first) {
            // SINGLE mode, from and to primitives in backedge are connected directly
            backedge_memory_mappings.emplace_back(
                backedge_from_prim, backedge_to_prim, initial_mem, body_network->get_stream());
            GPU_DEBUG_LOG << idx << ") add back_edge mapping with SINGLE type, initial_mem(" << initial_mem << ")" << std::endl;
        } else {
            // SINGLE_SHARED mode
            memory::ptr backedge_mem;
            auto output_prim = body_network->get_primitive(back_edge.from);

            if (is_dynamic()) {
                if (output_prim->outputs_allocated()) {
                    auto internal_output_prim_mem = output_prim->output_memory_ptr();
                    if (internal_output_prim_mem->get_layout() == initial_mem->get_layout()) {
                        backedge_mem = internal_output_prim_mem;
                        body_network->set_input_data(back_edge.to, backedge_mem);
                        GPU_DEBUG_LOG << idx << ") Get backedge_mem(" << backedge_mem
                                    << ") from back_edge.from(" << back_edge.from << ")" << std::endl;
                    } else {
                        // When input layout is changed or backedge_mem is null
                        // because output layout of body network is not calculated yet,
                        // Set backedge_mem to nullptr and update it after first execution.
                        body_network->set_input_data(back_edge.to, initial_mem);
                        GPU_DEBUG_LOG << idx << ") Just set input data using initial_mem because back_edge.from("
                                                << back_edge.from << ") layout is changed or backedge_mem is nullptr" << std::endl;
                    }
                } else {
                    body_network->set_input_data(back_edge.to, initial_mem);
                    GPU_DEBUG_LOG << idx << ") Just set input data using initial_mem because back_edge.from("
                                            << back_edge.from << ") has dynamic layout now" << std::endl;
                }
            } else {
                if (output_mapping.empty()) {
                    backedge_mem = output_prim->output_memory_ptr();
                    body_network->set_input_data(back_edge.to, backedge_mem);
                    GPU_DEBUG_LOG << idx << ") Get backedge_mem(" << backedge_mem
                                    << ") from back_edge.from(" << back_edge.from << ")" << std::endl;
                } else {
                    // Set input and output memory for body_network using external output memory of loop op
                    auto& out_mapping_ext_id = output_mapping.front()->external_id;
                    backedge_mem = get_external_memory(out_mapping_ext_id.pid, out_mapping_ext_id.idx);
                    GPU_DEBUG_LOG << idx << ") Get backedge_mem(" << backedge_mem << ") from output_mapping_external_id.pid("
                                    << out_mapping_ext_id.pid << ")" << std::endl;

                    body_network->set_input_data(back_edge.to, backedge_mem);
                    body_network->set_output_memory(back_edge.from, backedge_mem);
                }
            }

            backedge_memory_mappings.emplace_back(
                backedge_from_prim, backedge_to_prim, backedge_mem, initial_mem, body_network->get_stream());
            GPU_DEBUG_LOG << idx << ") add back_edge mapping with SINGLE_SHARED type, backedge_mem("
                            << backedge_mem << "), initial_mem(" << initial_mem << ")" << std::endl;
        }
    }
}

std::shared_ptr<loop_inst::concatenated_memory_mapping> loop_inst::get_sliced_mem(const primitive_id& internal_id) const {
    for (const auto& mem_mapping : concatenated_input_mem_mappings) {
        if (mem_mapping->sliced_data_prim->id() == internal_id) {
            return mem_mapping;
        }
    }
    for (const auto& mem_mapping : concatenated_output_mem_mappings) {
        if (mem_mapping->sliced_data_prim->id() == internal_id) {
            return mem_mapping;
        }
    }
    return nullptr; // not found
}

void loop_inst::validate_backedges(loop_node const & node) const {
    const auto& back_edges = node.get_back_edges();
    const auto& input_primitive_maps = node.get_input_primitive_maps();

    // check input with iteration axis has backedge
    for (const auto& back_edge : back_edges) {
        for (const auto& mapping : input_primitive_maps) {
            OPENVINO_ASSERT((mapping.internal_id.pid != back_edge.to || mapping.axis < 0),
                node.id(), ": input with iteration axis should not have backedges");
        }
    }
}

memory::ptr loop_inst::get_external_memory(const primitive_id& external_id, size_t mem_idx) const {
    const auto outputPrim = _network.get_primitive(external_id);
    if (outputPrim->outputs_allocated()) {
        return outputPrim->output_memory_ptr(mem_idx);
    }
    return nullptr;
}

layout loop_inst::get_external_output_layout(const primitive_id& external_id, size_t mem_idx) const {
    const auto outputPrim = _network.get_primitive(external_id);
    return outputPrim->get_output_layout(mem_idx);
}

loop_inst::typed_primitive_inst(network & network, loop_node const & node)
    : parent(network, node),
        preproc_memories_done(false),
        body_network(network::allocate_network(network.get_stream_ptr(),
                                                node.get_body_program(),
                                                false,
                                                network.is_primary_stream())) {
    const primitive_id& num_iterations_id = node.get_num_iterations_id();
    OPENVINO_ASSERT(node.get_program().get_node(num_iterations_id).is_type<mutable_data>(),
                        node.id(), ": num_iterations is not mutable_data");
    OPENVINO_ASSERT(check_if_axis_is_set_properly(node), node.id(), ": axis is not set properly");

    set_inner_networks({body_network});
    validate_backedges(node);
    validate_mappings(node);

    _input_primitive_maps = node.get_input_primitive_maps();
    _output_primitive_maps = node.get_output_primitive_maps();
    _back_edges = node.get_back_edges();
    _trip_count_id = node.get_trip_count_id();
    _initial_execution_id = node.get_initial_execution_id();
    _current_iteration_id = node.get_current_iteration_id();
    _condition_id = node.get_execution_condition_id();
    _num_iterations_id = node.get_num_iterations_id();
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
    ob << _num_iterations_id;
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
    ib >> _num_iterations_id;
    body_network = std::make_shared<cldnn::network>(ib, get_network().get_stream_ptr(), get_network().get_engine(), get_network().is_primary_stream(), 0);
}

void loop_inst::postprocess_output_memory(bool is_dynamic) {
    if (is_dynamic) {
        std::vector<cldnn::memory::ptr> external_outputs;
        external_outputs.resize(outputs_memory_count());
        for (size_t i = 0; i < _output_primitive_maps.size(); ++i) {
            const auto& output_mapping = _output_primitive_maps.at(i);
            const auto& external_id = output_mapping.external_id;
            const auto& internal_id = output_mapping.internal_id;
            bool output_allocated = (static_cast<size_t>(external_id.idx) < _outputs.size() && _outputs[external_id.idx] != nullptr);
            if (output_mapping.axis < 0) {
                auto internalOutputPrim = get_body_network()->get_primitive(internal_id.pid);
                auto internal_mem = internalOutputPrim->output_memory_ptr(internal_id.idx);
                OPENVINO_ASSERT(internal_mem != nullptr, "internal_mem should not be nullptr");
                if (!output_allocated) {
                    external_outputs[external_id.idx] = internal_mem;
                } else {
                    auto external_mem = _outputs[external_id.idx];
                    if (external_mem->get_layout() != internal_mem->get_layout()) {
                        external_outputs[external_id.idx] = internal_mem;
                    } else if (external_mem != internal_mem) {
                        external_mem->copy_from(get_network().get_stream(), *internal_mem);
                        external_outputs[external_id.idx] = external_mem;
                    }
                }
            } else {
                if (!output_allocated || shape_changed()) {
                    auto concat_layout = _impl_params->get_output_layout(external_id.idx);
                    auto concat_mem = _network.get_engine().allocate_memory(concat_layout, 0);
                    external_outputs[external_id.idx] = concat_mem;
                    auto iter = std::find_if(concatenated_output_mem_mappings.begin(),
                                                concatenated_output_mem_mappings.end(),
                                                [&](std::shared_ptr<loop_inst::concatenated_memory_mapping> &concat_output){
                                                    return concat_output->concat_data_id == external_id;
                                                });
                    if (iter != concatenated_output_mem_mappings.end()) {
                        (*iter)->update_concatenated_mem(concat_mem);
                    }
                } else {
                    external_outputs[external_id.idx] = _outputs[external_id.idx];
                }
            }
        }
        _outputs = external_outputs;
    }

    for (size_t i = 0; i < concatenated_output_mem_mappings.size(); ++i) {
        const auto& concat_output = concatenated_output_mem_mappings.at(i);
        concat_output->restore_concatenated_mem();
    }
}

void loop_inst::reset_memory() {
    backedge_memory_mappings.clear();
    concatenated_input_mem_mappings.clear();
    for (auto concat_mem_map : concatenated_output_mem_mappings) {
        concat_mem_map->reset_data_for_shape_changed();
    }
}


void loop_inst::update_output_layout() {
    if (_node == nullptr)
        return;

    auto memory_deps = _node->get_const_memory_deps();
    for (auto& i : _node->get_shape_infer_dependencies()) {
        auto dep_id = _node->get_dependency(i).id();
        if (memory_deps.count(i) > 0 || i >= _node->get_dependencies().size()) {
            continue;
        }

        auto dep_mem = _network.get_output_memory(dep_id);
        memory_deps.insert({i, dep_mem});
    }
    _impl_params->memory_deps = memory_deps;

    auto new_layouts = _node->type()->calc_output_layouts(*_node, *_impl_params);
    if (new_layouts.empty()) {
        auto new_layout = _node->type()->calc_output_layout(*_node, *_impl_params);
        new_layout.data_padding = padding::max(_node->get_primitive()->output_paddings[0], new_layout.data_padding);
        _impl_params->output_layouts[0] = new_layout;
    } else {
        if (_impl_params->output_layouts.size() < new_layouts.size()) {
            _impl_params->output_layouts.resize(new_layouts.size());
        }
        for (size_t i = 0; i < new_layouts.size(); ++i) {
            auto new_layout = new_layouts[i];
            new_layout.data_padding = padding::max(_node->get_primitive()->output_paddings[i], new_layout.data_padding);
            _impl_params->output_layouts[i] = new_layout;
        }
    }
}
}  // namespace cldnn
