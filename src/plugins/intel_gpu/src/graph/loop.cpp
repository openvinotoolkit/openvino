// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "loop_inst.h"

#include "data_inst.h"
#include "mutable_data_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include <string>
#include <exception>
#include <algorithm>
#include "openvino/reference/concat.hpp"
#include "openvino/reference/split.hpp"
#include "openvino/reference/utils/coordinate_range.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(loop)

std::map<size_t, memory::ptr> loop_node::get_memory_deps() const {
    auto memory_deps = get_const_memory_deps();
    for (auto& i : get_shape_infer_dependencies()) {
        auto& dep = get_dependency(i);
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
        } else {
            // if num_iterations is zero, it means loop does not run inner body network.
            // in the case of dynamic output layout, dynamic dimension will be replaced to zero.
            if (num_iterations == 0) {
                auto shape = loop_output_layout.get_partial_shape();
                for (size_t i = 0; i < shape.size(); i++) {
                    if (shape[i].is_dynamic())
                        shape[i] = 0;
                }
                loop_output_layout.set_partial_shape(shape);
            }
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
        output_layouts = get_output_layouts<program_node*>(impl_param, body_outputs, prim->max_num_iterations);
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
                    if (mem_mapping->get_sliced_data_prim_id() == input_map->internal_id.pid) {
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
                    if (mem_mapping->get_sliced_data_prim_id() == internal_id) {
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
                        if (backedge_to_prim.get() == backedge_from_prim->dependencies().front().first) {
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

loop_inst::concatenated_memory_mapping::ptr loop_inst::create_concat_memory_map(const cldnn::loop::io_primitive_map& io_prim_map,
                                                                                    memory::ptr extern_mem_ptr,
                                                                                    const int64_t num_iterations) {
    OPENVINO_ASSERT(io_prim_map.axis >= 0, "axis should not be negative");
    const auto& external_id = io_prim_map.external_id;
    const auto& internal_id = io_prim_map.internal_id;
    auto& engine = body_network->get_engine();
    auto& stream = body_network->get_stream();
    auto intern_prim = body_network->get_primitive(internal_id.pid);
    auto extern_prim = get_network().get_primitive(external_id.pid);

    std::vector<memory::ptr> sliced_mems;

    // if memory is nullptr, that means memory is not allocated yet because current network is dynamic shape model.
    // In dynamic model, we can't calculate num_element_iteration, start, and sliced_layout.
    // will recalculate that parameters in backedge preprocessing map after first execution.
    if (extern_mem_ptr != nullptr) {
        layout sliced_layout = intern_prim->get_output_layout(internal_id.idx);
        auto inter_mem_ptr = intern_prim->output_memory_ptr(internal_id.idx);
        if (inter_mem_ptr == nullptr || get_flag(ExecutionFlags::SHAPE_CHANGED)) {
            // if inner body intern_prim has no output memory because it has dynamic shape,
            // calculate inner body intern_prim layout using concat_mem's layout.
            auto updated_sliced_layout = sliced_layout.get_partial_shape();
            OPENVINO_ASSERT(updated_sliced_layout[io_prim_map.axis].is_static() || num_iterations > 0,
                                    "Not allowed dynamic dimension for axis when num_iteraiont is negative");

            auto origin_input_layout = body_network->get_primitive(internal_id.pid)->get_node_output_layout();
            auto concat_pshape = extern_prim->get_output_layout().get_partial_shape();
            const auto shape_size = concat_pshape.size();
            if (origin_input_layout.is_dynamic()) {
                auto origin_input_pshape = origin_input_layout.get_partial_shape();
                for (size_t i = 0; i < shape_size; i++) {
                    if (origin_input_pshape[i].is_dynamic()) {
                        updated_sliced_layout[i] = concat_pshape[i];
                    }
                }
            }
            GPU_DEBUG_LOG << "output pshape for [" << intern_prim->id() << "] is changed from "
                            << sliced_layout.get_partial_shape().to_string()
                            << " to " << updated_sliced_layout.to_string() << std::endl;
            sliced_layout.set_partial_shape(updated_sliced_layout);
            inter_mem_ptr = engine.allocate_memory(sliced_layout);
            intern_prim->set_output_layout(sliced_layout, internal_id.idx);
        }

        // When num_iterations is -1, allocate first sliced_mem and allocate sliced memory if additional sliced mem is required
        if (num_iterations < 0) {
            sliced_mems.push_back(inter_mem_ptr);
        } else {
            sliced_mems.reserve(num_iterations);
            sliced_mems.push_back(inter_mem_ptr);
            for (int j = 1; j < num_iterations; ++j) {
                memory::ptr sliced_mem = engine.allocate_memory(sliced_layout);
                sliced_mems.push_back(sliced_mem);
            }
        }
    }
    auto sliced_data_prim = body_network->get_primitive(internal_id.pid);
    auto concat_data_prim = get_network().get_primitive(external_id.pid);
    auto concat_data_id   = external_id;
    return std::make_shared<concatenated_memory_mapping>(extern_mem_ptr, sliced_mems, stream, engine,
                                                concat_data_prim, sliced_data_prim, io_prim_map);
}

void loop_inst::preprocess_output_memory(const int64_t num_iterations) {
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
                    return concat_mem_map->get_sliced_data_prim_id() == internal_id.pid;
                });
            if (iter == concatenated_output_mem_mappings.end()) {
                auto memory_mapping_info = create_concat_memory_map(output_mapping, memory, num_iterations);
                concatenated_output_mem_mappings.push_back(memory_mapping_info);
                GPU_DEBUG_LOG << i << ") generate concat output memory mapping: " << memory_mapping_info->to_string() << std::endl;
            } else {
                GPU_DEBUG_LOG << i << ") memory_mapping_info is already existed : " << (*iter)->to_string() << std::endl;
            }
        }
    }
}

void loop_inst::preprocess_input_memory(const int64_t num_iterations) {
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

            if (input_map->axis < 0) {
                auto input_inst = body_network->get_primitive(internal_id.pid);
                if (!input_inst->get_output_layout().identical(_impl_params->get_input_layout(memory_num))) {
                    input_inst->set_output_layout(_impl_params->get_input_layout(memory_num));
                }

                if (!input_inst->get_output_layout().is_dynamic() && !memory->get_layout().identical(input_inst->get_output_layout())) {
                    OPENVINO_ASSERT(input_inst->get_output_layout().bytes_count() <= memory->get_layout().bytes_count(),
                                    "input layout size(", input_inst->get_output_layout().to_short_string(),
                                    ") should not exceed memory size(", memory->get_layout().to_short_string(), ")");
                    memory = body_network->get_engine().reinterpret_buffer(*memory, input_inst->get_output_layout());
                    GPU_DEBUG_LOG << input_inst->id() << " is changed memory because layout is changed from "
                                        << memory->get_layout().to_short_string()
                                        << " to " << input_inst->get_output_layout().to_short_string() << std::endl;
                }

                auto internal_input_memory = memory;
                auto iter = std::find_if(_back_edges.begin(), _back_edges.end(), [&](cldnn::loop::backedge_mapping& mapping) {
                    return (mapping.to == internal_id.pid);
                });
                // if internal input memory is in backedge, allocate new memory.
                // Because internal input memory's data will be updated through backedge process.
                if (iter != _back_edges.end()) {
                    internal_input_memory = body_network->get_engine().allocate_memory(memory->get_layout(), false);
                    internal_input_memory->copy_from(body_network->get_stream(), *memory);
                    GPU_DEBUG_LOG << "Input memory of internal node(" << internal_id.to_string() << ") is set to new memory("
                                    << internal_input_memory << ", " << internal_input_memory->get_layout().to_short_string()
                                    << ") instead of external node(" << external_id.to_string()
                                    <<")'s memory(" << memory << "," << memory->get_layout().to_short_string() << ")" << std::endl;
                }

                body_network->set_input_data(internal_id.pid, internal_input_memory);
            } else {
                OPENVINO_ASSERT(memory != nullptr, "In preprocessing concat input mapping, concat memory should be allocated");
                auto memory_mapping_info = create_concat_memory_map(*input_map, memory, num_iterations);
                concatenated_input_mem_mappings.push_back(memory_mapping_info);
                GPU_DEBUG_LOG << i << ") generate concat input memory mapping: " << memory_mapping_info->to_string() << std::endl;
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
        // in case where memory buffer has been over-allocated by shape predictor, memory layout might be unexpected shape.
        // so memory layout needs to be re-interprete according to original layout.
        auto initial_layout = get_external_output_layout(external_id.pid, external_id.idx);
        OPENVINO_ASSERT(initial_mem != nullptr, "initial_mem should not be null");
        if (!initial_mem->get_layout().identical(initial_layout)) {
            OPENVINO_ASSERT(initial_layout.bytes_count() <= initial_mem->get_layout().bytes_count(),
                            "initial layout size(", initial_layout.to_short_string(),
                            ") should not exceed initial memory size(", initial_mem->get_layout().to_short_string(), ")");
            initial_mem = body_network->get_engine().reinterpret_buffer(*initial_mem, initial_layout);
        }

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
        // Set backedge mode to SINGLE when backedge_from_prim has multiple users.
        } else if ((output_mapping.empty() && backedge_to_prim.get() == backedge_from_prim->dependencies().front().first)
                || (backedge_to_prim->get_users().size() > 1) ) {
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
        if (mem_mapping->get_sliced_data_prim_id() == internal_id) {
            return mem_mapping;
        }
    }
    for (const auto& mem_mapping : concatenated_output_mem_mappings) {
        if (mem_mapping->get_sliced_data_prim_id() == internal_id) {
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
                node.id(), ": input with iteration axis should not have backedges external_id: ",
                mapping.external_id.to_string(), ", internal_id: ", mapping.internal_id.to_string(),
                ", back_edge.to: ", back_edge.to, ", back_edge.from ", back_edge.from,
                ", mapping.axis: ", std::to_string(mapping.axis));
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

void loop_inst::postprocess_output_memory(bool is_dynamic, int64_t current_iteration) {
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
                    GPU_DEBUG_LOG << "[Internal: " << internal_id.to_string() << ", External: " << external_id.to_string() << " ] "
                                    << "Set internal memory(" << internal_mem << ") to external output because external output memory is nullptr." << std::endl;
                } else {
                    auto external_mem = _outputs[external_id.idx];
                    if (external_mem != internal_mem) {
                        if (external_mem->get_layout() != internal_mem->get_layout()) {
                            external_outputs[external_id.idx] = internal_mem;
                            GPU_DEBUG_LOG << "[Internal: " << internal_id.to_string() << ", External: " << external_id.to_string() << " ] "
                                            << "Set internal memory(" << internal_mem
                                            << ") to external output for different layout between external_mem and internal_mem." << std::endl;
                        } else {
                            external_mem->copy_from(get_network().get_stream(), *internal_mem);
                            external_outputs[external_id.idx] = external_mem;
                            GPU_DEBUG_LOG << "[Internal: " << internal_id.to_string() << ", External: " << external_id.to_string() << " ] "
                                            << "Copy internal memory data to external memory data." << std::endl;
                        }
                    } else {
                        external_outputs[external_id.idx] = external_mem;
                        GPU_DEBUG_LOG << "[Internal: " << internal_id.to_string() << ", External: " << external_id.to_string() << " ] "
                                        << " Have same memory pointer." << std::endl;
                    }
                }
            } else {
                if (!output_allocated || get_flag(ExecutionFlags::SHAPE_CHANGED)) {
                    auto concat_layout = _impl_params->get_output_layout(external_id.idx);
                    auto concat_mem = _network.get_engine().allocate_memory(concat_layout, false);
                    external_outputs[external_id.idx] = concat_mem;
                    auto iter = std::find_if(concatenated_output_mem_mappings.begin(),
                                                concatenated_output_mem_mappings.end(),
                                                [&](std::shared_ptr<loop_inst::concatenated_memory_mapping> &concat_output){
                                                    return concat_output->get_external_id() == external_id;
                                                });
                    if (iter != concatenated_output_mem_mappings.end()) {
                        (*iter)->update_concatenated_mem(concat_mem);
                        GPU_DEBUG_LOG << "[Internal: " << internal_id.to_string() << ", External: " << external_id.to_string() << " ]"
                                        << " Update concat_mem" << std::endl;
                    }
                    GPU_DEBUG_IF(iter == concatenated_output_mem_mappings.end()) {
                        GPU_DEBUG_LOG << "[Internal: " << internal_id.to_string() << ", External: " << external_id.to_string() << " ]"
                                        << " Can't find concatenated_memory_mapping" << std::endl;
                    }
                } else {
                    external_outputs[external_id.idx] = _outputs[external_id.idx];
                    GPU_DEBUG_LOG << "[Internal: " << internal_id.to_string() << ", External: " << external_id.to_string() << " ]"
                                    << " No update concat_mem" << std::endl;
                }
            }
        }
        _outputs = external_outputs;
    }

    for (size_t i = 0; i < concatenated_output_mem_mappings.size(); ++i) {
        const auto& concat_output = concatenated_output_mem_mappings.at(i);
        concat_output->concat_mem(current_iteration);
    }
}

void loop_inst::reset_memory() {
    GPU_DEBUG_LOG << "Reset memory" << std::endl;
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
        new_layout.data_padding = padding::max(_node->get_primitive()->get_output_padding(0), new_layout.data_padding);
        _impl_params->output_layouts[0] = new_layout;
    } else {
        if (_impl_params->output_layouts.size() < new_layouts.size()) {
            _impl_params->output_layouts.resize(new_layouts.size());
        }
        for (size_t i = 0; i < new_layouts.size(); ++i) {
            auto new_layout = new_layouts[i];
            new_layout.data_padding = padding::max(_node->get_primitive()->get_output_padding(i), new_layout.data_padding);
            _impl_params->output_layouts[i] = new_layout;
        }
    }
}

void loop_inst::concatenated_memory_mapping::slice_mem(const int64_t num_iterations) const {
    size_t num_iters = static_cast<size_t>(num_iterations);
    OPENVINO_ASSERT(num_iters > 0 && num_iters == sliced_mems.size(), "num_iterations(", num_iters,
                            ") should be same with sliced_mems.size(", sliced_mems.size(), ")");
    OPENVINO_ASSERT(concatenated_mem != nullptr, "concatenated_mem should not be nullptr");

    auto concat_layout = concat_data_prim->get_output_layout(io_prim_map.external_id.idx);
    auto sliced_layout = sliced_data_prim->get_output_layout(io_prim_map.internal_id.idx);
    auto concat_mem_shape = concat_layout.get_shape();
    auto sliced_mem_shape = sliced_layout.get_shape();
    auto elem_size = ov::element::Type(concat_layout.data_type).size();
    const auto stride = io_prim_map.stride;
    const auto axis = io_prim_map.axis;
    const auto step = std::abs(stride);
    OPENVINO_ASSERT((static_cast<size_t>(step) == sliced_mem_shape[axis])
                        && (concat_mem_shape[axis] >= num_iterations * sliced_mem_shape[axis]),
                        "slice_mem: concat_mem_shape[axis(", axis, "),step(", step, ")](",
                        concat_mem_shape.to_string(), ") != num_iterations(",
                        num_iterations, ") * sliced_mem_shape[axis](", sliced_mem_shape.to_string(), ")");
    std::vector<char*> pointers_to_data(num_iters);
    for (size_t i = 0; i < num_iters; i++) {
        auto mem = sliced_mems[i];
        pointers_to_data[stride > 0 ? i : (num_iters - i - 1)] = reinterpret_cast<char*>(mem->lock(stream));
    }

    char* concat_data = reinterpret_cast<char*>(concatenated_mem->lock(stream, cldnn::mem_lock_type::read));
    auto dims = concat_mem_shape.size();
    if (!format::is_default_format(concat_layout.format) || dims == 1 || concat_layout.data_padding) {
        // BE CAREFUL: ov::reference::split is extremely slow.
        // If we encounter any case where this code path is executed, we need to optimize it
        ov::reference::split(concat_data, concat_mem_shape, elem_size, axis, num_iters, pointers_to_data.data());
    } else {
        const size_t part_length = concat_mem_shape.at(axis) / num_iters;
        const size_t inner_axis = axis + 1;
        auto output_shape = concat_mem_shape;
        auto out_data = pointers_to_data.data();
        output_shape[axis] = part_length;

        ov::Coordinate lower_bounds(concat_mem_shape.size(), 0);
        ov::Coordinate upper_bounds = output_shape;
        auto& lb_at_axis = lower_bounds[axis];
        auto& ub_at_axis = upper_bounds[axis];

        // Format of concat_layout is invalid here : No mixed order
        size_t continuous_size = 1;
        for (auto iter = inner_axis ; iter < dims ; ++iter) {
            continuous_size *= ((output_shape.size() > iter) ? output_shape[iter] : 1);
        }

        // Set stride values of inner axes to get a continuous copy size
        auto strides = ov::Strides(lower_bounds.size(), 1);
        for (size_t iter = inner_axis; iter < dims ; ++iter)
            strides[iter] = upper_bounds[iter];

        const auto strides_copy_size = elem_size * continuous_size;
        const auto out_last = std::next(out_data, num_iters);
        for (auto out_iter = out_data; out_iter != out_last; ++out_iter) {
            auto dst_mem = *out_iter;
            auto slice_ranges = ov::coordinates::slice(concat_mem_shape, lower_bounds, upper_bounds, strides);
            for (const auto& range : slice_ranges) {
                const auto src_mem = concat_data + range.begin_index * elem_size;
                std::memcpy(dst_mem, src_mem, strides_copy_size);
                std::advance(dst_mem, strides_copy_size);
            }

            lb_at_axis += part_length;
            ub_at_axis += part_length;
        }
    }

    for (size_t i = 0; i < num_iters; i++) {
        sliced_mems[i]->unlock(stream);
    }
    concatenated_mem->unlock(stream);

    GPU_DEBUG_LOG << "slice memory [" << io_prim_map.to_short_string() << "] from concat_mem["
                    << concatenated_mem->get_layout().to_short_string()
                    << "], current_iteration: " << num_iterations << ", stride: " << stride
                    << " to sliced_mems[" << sliced_mems.front()->get_layout().to_short_string() << "]" << std::endl;
}

void loop_inst::concatenated_memory_mapping::concat_mem(const int64_t curent_iterations) const {
    size_t curr_iters = static_cast<size_t>(curent_iterations);
    OPENVINO_ASSERT(sliced_mems.size() >= curr_iters, "curent_iterations(", curr_iters,
                        ") should be less than the number of sliced_mems(", sliced_mems.size(), ")");
    OPENVINO_ASSERT(concatenated_mem != nullptr, "concatenated_mem should not be nullptr");

    auto concat_layout = concat_data_prim->get_output_layout(io_prim_map.external_id.idx);
    auto sliced_layout = sliced_data_prim->get_output_layout(io_prim_map.internal_id.idx);
    auto concat_mem_shape = concat_layout.get_shape();
    auto sliced_mem_shape = sliced_layout.get_shape();
    auto elem_size = ov::element::Type(concat_layout.data_type).size();
    const auto stride = io_prim_map.stride;
    const auto axis = io_prim_map.axis;
    const auto step = std::abs(stride);
    OPENVINO_ASSERT((static_cast<size_t>(step) == sliced_mem_shape[axis])
                        && (concat_mem_shape[axis] >= curent_iterations * sliced_mem_shape[axis]),
                        "concat_mem: concat_mem_shape[axis(", axis, "),step(", step, ")](",
                        concat_mem_shape.to_string(), ") != curent_iterations(",
                        curent_iterations, ") * sliced_mem_shape[axis](", sliced_mem_shape.to_string(), ")");
    std::vector<ov::Shape> shapes_to_concat(curr_iters, sliced_mem_shape);
    std::vector<const char*> pointers_to_data(curr_iters);
    for (size_t i = 0; i < curr_iters; i++) {
        auto mem = sliced_mems[i];
        pointers_to_data[stride > 0 ? i : (curr_iters - i - 1)] = reinterpret_cast<const char*>(mem->lock(stream));
    }

    char* concat_data = reinterpret_cast<char*>(concatenated_mem->lock(stream));
    ov::reference::concat(pointers_to_data, concat_data, shapes_to_concat, concat_mem_shape, axis, elem_size);

    for (size_t i = 0; i < curr_iters; i++) {
        sliced_mems[i]->unlock(stream);
    }
    concatenated_mem->unlock(stream);
    GPU_DEBUG_LOG << "concatenate memory [" << io_prim_map.to_short_string() << "] from sliced_mems["
                    << sliced_mems.front()->get_layout().to_short_string() << "], current_iteration: "
                    << curent_iterations << ", stride: " << stride << " to concat_mem["
                    << concatenated_mem->get_layout().to_short_string() << "]" << std::endl;
}

int64_t loop_inst::calculate_num_iterations(const cldnn::loop::io_primitive_map& io_prim_map,
                                                ov::PartialShape& pshape) {
    OPENVINO_ASSERT(io_prim_map.stride != 0, "stride should not be zero");
    const auto space = pshape[io_prim_map.axis].get_length();
    const auto start = (io_prim_map.start < 0? (space + 1) : 0) + io_prim_map.start;
    const auto end   = (io_prim_map.end < 0? (space + 1) : 0) + io_prim_map.end;
    const auto step  = std::abs(io_prim_map.stride);
    const auto src   = io_prim_map.stride < 0 ? end : start;
    const auto dst   = io_prim_map.stride < 0 ? start : end;
    const auto len   = dst - src;
    OPENVINO_ASSERT(src >= 0 && dst > src && dst <= space && len >= static_cast<long>(step),
                        "invalid values in an iteration component start:",
                        io_prim_map.start, ", end: ", io_prim_map.end, ", stride:",
                        io_prim_map.stride, ", axis: ", io_prim_map.axis, ", dst: ",
                        dst, ", src: ", src, ", space: ", space, ", len: ",
                        len, ", step: ", step, ", pshape: ", pshape.to_string());
    OPENVINO_ASSERT(len % step == 0, "Each iteration should have same size: length(", len, ") % step(", step, ")");
    int64_t num_iterations = static_cast<int64_t>(len / step);
    {
        GPU_DEBUG_LOG << "Caculate num_iterations ..." << std::endl;
        GPU_DEBUG_LOG << "* io_prim_map.{start:" << io_prim_map.start << ", end:" << io_prim_map.end
                << ", stride: " << io_prim_map.stride << ", axis: " << io_prim_map.axis << "}" << std::endl;
        GPU_DEBUG_LOG << "* pshape : " << pshape.to_string() << std::endl;
        GPU_DEBUG_LOG << "* space  : " << space    << std::endl;
        GPU_DEBUG_LOG << "* start  : " << start    << std::endl;
        GPU_DEBUG_LOG << "* end    : " << end      << std::endl;
        GPU_DEBUG_LOG << "* step   : " << step     << std::endl;
        GPU_DEBUG_LOG << "* src    : " << src      << std::endl;
        GPU_DEBUG_LOG << "* dst    : " << dst      << std::endl;
        GPU_DEBUG_LOG << "* len    : " << len      << std::endl;
        GPU_DEBUG_LOG << "* num_iterations    : " << num_iterations << std::endl;
    }
    return num_iterations;
}

int64_t loop_inst::get_num_iterations() {
    int64_t num_iterations = -1;
    bool is_default_num_iter = true;
    for (auto& input_map : _input_primitive_maps) {
        if (input_map.axis == -1)
            continue;
        const auto& external_id = input_map.external_id;
        auto exteranl_input_inst = get_network().get_primitive(external_id.pid);
        auto concat_shape = exteranl_input_inst->get_output_layout(external_id.idx).get_partial_shape();

        if (concat_shape[input_map.axis].get_length() == 0)
            continue;

        const auto current_num_iterations = calculate_num_iterations(input_map, concat_shape);
        if (is_default_num_iter) {
            is_default_num_iter = false;
            num_iterations = current_num_iterations;
        }
        OPENVINO_ASSERT(num_iterations == current_num_iterations,
                            "iteration num shuld be same between ", num_iterations, " and ", current_num_iterations);
    }

    for (auto& output_map : _output_primitive_maps) {
        if (output_map.axis == -1)
            continue;

        const auto& external_id = output_map.external_id;
        auto exteranl_output_inst = get_network().get_primitive(external_id.pid);
        auto concat_shape = exteranl_output_inst->get_output_layout(external_id.idx).get_partial_shape();

        if (concat_shape[output_map.axis].is_dynamic() || concat_shape[output_map.axis].get_length() == 0)
            continue;

        const auto current_num_iterations = calculate_num_iterations(output_map, concat_shape);
        if (is_default_num_iter) {
            is_default_num_iter = false;
            num_iterations = current_num_iterations;
        }
        // only check num_terations when shape is not changed.
        if (preproc_memories_done)
            OPENVINO_ASSERT(num_iterations == current_num_iterations,
                            "iteration num shuld be same between ", num_iterations, " and ", current_num_iterations);
    }
    return num_iterations;
}

void loop_inst::set_memory_in_body_network(cldnn::network::ptr body_network,
                const std::shared_ptr<cldnn::primitive_inst>& inst, memory::ptr mem) {
    if (inst->is_input()) {
        // in case where memory buffer has been over-allocated by shape predictor, memory layout might be unexpected shape.
        // so memory layout needs to be re-interpret according to original layout.
        memory::ptr updated_mem = mem;
        layout impl_layout = inst->get_impl_params()->get_output_layout();
        OPENVINO_ASSERT(impl_layout.bytes_count() <= updated_mem->get_layout().bytes_count(),
                        "impl_params layout size(", impl_layout.to_short_string(),
                        ") should not exceed memory size(", updated_mem->get_layout().to_short_string(), ")");
        // Set need_to_check_memory_to_set to false to set output memory even if the input node has static shape,
        body_network->set_input_data(inst->id(), updated_mem, false);
        // Update impl_params.output_layouts[0] to updated_mem's layout
        inst->update_shape();
    } else if (inst->is_output()) {
        body_network->set_output_memory(inst->id(), mem);
    } else {
        inst->set_output_memory(mem, false);
    }
}

std::vector<event::ptr> loop_inst::handle_buffers_for_next_iteration(const loop_inst::backedge_memory_mapping& mapping,
                                                    network::ptr body_network, int64_t iter) {
    std::vector<event::ptr> event_vec;
    OPENVINO_ASSERT(iter >= 0, "iteration should not be negative : ", iter);
    if (mapping.type == loop_inst::backedge_memory_mapping::CONCAT_OUTPUT) {
        if (iter == 0) {
            set_memory_in_body_network(body_network, mapping.to_primitive, mapping.initial_mem);
            GPU_DEBUG_LOG << iter << ") [CONCAT_OUTPUT] Copy data from inintal_mem(" << mapping.initial_mem
                            << ") to " << mapping.to_primitive->id() << std::endl;
        } else if (iter > 0) {
            if (is_dynamic()) {
                auto from_id = mapping.from_primitive->id();
                if (body_network->has_event(from_id)) {
                    auto ev = body_network->get_primitive_event(from_id);
                    if (ev) ev->wait();
                }
                // In dynamic model, just copy data from inner body output to inner body input in back_edges.
                memory::ptr to_mem = mapping.to_primitive->output_memory_ptr();
                memory::ptr from_mem = mapping.from_primitive->output_memory_ptr();
                auto ev = to_mem->copy_from(body_network->get_stream(), *(from_mem));
                if (ev) event_vec = {ev};
                GPU_DEBUG_LOG << iter << ") [CONCAT_OUTPUT] Copy data from [" << mapping.from_primitive->id() << "(" << from_mem
                                << ")] to [" << mapping.to_primitive->id() << "(" << to_mem << ")]" << std::endl;
            } else {
                auto mem = mapping.concat_mem_mapping->get_sliced_mems().at(iter - 1);
                set_memory_in_body_network(body_network, mapping.to_primitive, mem);
                GPU_DEBUG_LOG << iter << ") [CONCAT_OUTPUT] Set memory from concat_mem[" << (iter - 1) << "](" << mem
                                << ") to " << mapping.to_primitive->id() << ")" << std::endl;
            }
        }
    } else if (mapping.type ==  loop_inst::backedge_memory_mapping::SINGLE_SHARED) {
        if (iter == 0) {
            if (mapping.from_mem != nullptr) {
                auto ev = mapping.from_mem->copy_from(body_network->get_stream(), *(mapping.initial_mem));
                if (ev) event_vec = {ev};
                GPU_DEBUG_LOG << iter << ") [SINGLE_SHARED] Copy data from inintal_mem(" << mapping.initial_mem << ")" << std::endl;
            }
        } else {
            // In dynamic model, output memory is not defined before execution.
            // After body network execution, replace input memory from initial_mem(external input memory) to output memory.
            if (mapping.from_mem == nullptr) {
                mapping.from_mem = mapping.from_primitive->output_memory_ptr();
                OPENVINO_ASSERT(mapping.from_mem != nullptr, "from_mem should not be null");
                set_memory_in_body_network(body_network, mapping.to_primitive, mapping.from_mem);
                GPU_DEBUG_LOG << iter << ") [SINGLE_SHARED] Set memory from from_mem(" << mapping.from_mem
                                << ") to " << mapping.to_primitive->id() << ")" << std::endl;
            }
        }
    } else if (mapping.type == loop_inst::backedge_memory_mapping::SINGLE) {
        memory::ptr to_mem = mapping.to_primitive->output_memory_ptr();

        if (is_dynamic()) {
            // In dynamic model, do not swap memory buffer between input and output in inner body network.
            // Check size of input buffer memory and output buffer memory
            // If size is differnet, allocate new input memory for the required size,
            // Else just copy data from input buffer memory to output buffer memory.
            cldnn::event::ptr ev;
            if (iter == 0) {
                auto to_id = mapping.to_primitive->id();
                // Check backedge_to shape needs to be updated by initial_mem
                OPENVINO_ASSERT(mapping.initial_mem != nullptr, "initial_mem should not be null");
                if (!mapping.initial_mem->get_layout().identical(to_mem->get_layout())) {
                    to_mem = body_network->get_engine().allocate_memory(mapping.initial_mem->get_layout(), false);

                    body_network->set_input_data(to_id, to_mem);
                    ev = to_mem->copy_from(body_network->get_stream(), *(mapping.initial_mem));
                    GPU_DEBUG_LOG << iter << ") [SINGLE] Backedge_to node(" << to_id << ") is set to new memory("
                                    << to_mem << ", " << to_mem->get_layout().to_short_string()
                                    << ") because of shape update from initial memory("
                                    << mapping.initial_mem << "," << mapping.initial_mem->get_layout().to_short_string() << ")" << std::endl;
                } else {
                    ev = to_mem->copy_from(body_network->get_stream(), *(mapping.initial_mem));
                    GPU_DEBUG_LOG << iter << ") [SINGLE] Copy data from inintal_mem(" << mapping.initial_mem << ")" << std::endl;
                }
            } else {
                auto from_id = mapping.from_primitive->id();
                auto to_id = mapping.to_primitive->id();
                if (body_network->has_event(from_id)) {
                    auto ev = body_network->get_primitive_event(from_id);
                    if (ev) ev->wait();
                }
                memory::ptr from_mem = mapping.from_primitive->output_memory_ptr();

                // Check backedge_to shape needs to be updated by backedge_from
                if (!from_mem->get_layout().identical(to_mem->get_layout())) {
                    to_mem = body_network->get_engine().allocate_memory(from_mem->get_layout(), false);
                    GPU_DEBUG_LOG << iter << ") [SINGLE] Backedge_to node(" << to_id << ") is set to new memory("
                                    << to_mem << ", " << to_mem->get_layout().to_short_string()
                                    << ") because of shape update from backedge_from()" << from_id
                                    <<")'s memory(" << from_mem << "," << from_mem->get_layout().to_short_string() << ")" << std::endl;
                    body_network->set_input_data(to_id, to_mem);
                    ev = to_mem->copy_from(body_network->get_stream(), *(from_mem));
                } else {
                    ev = to_mem->copy_from(body_network->get_stream(), *(from_mem));
                }
                GPU_DEBUG_LOG << iter << ") [SINGLE] Copy data from [" << mapping.from_primitive->id()
                            << "(" << from_mem << ")] to [" << mapping.to_primitive->id() << "(" << to_mem << ")]" << std::endl;
            }
            if (ev) event_vec = {ev};
        } else {
            if (iter == 0) {
                auto ev = to_mem->copy_from(body_network->get_stream(), *(mapping.initial_mem));
                if (ev) event_vec = {ev};
                GPU_DEBUG_LOG << iter << ") [SINGLE] Copy data from inintal_mem(" << mapping.initial_mem << ")" << std::endl;
            } else {
                // In static model, swap memory buffer between output and input in inner body network
                memory::ptr from_mem = mapping.from_primitive->output_memory_ptr();
                GPU_DEBUG_LOG << iter << ") [SINGLE] Before swap between [" << mapping.from_primitive->id()
                            << "(" << mapping.from_primitive->output_memory_ptr() << ")] and [" << mapping.to_primitive->id()
                            << "(" << mapping.to_primitive->output_memory_ptr() << ")]" << std::endl;
                set_memory_in_body_network(body_network, mapping.to_primitive, std::move(from_mem));
                set_memory_in_body_network(body_network, mapping.from_primitive, std::move(to_mem));
                GPU_DEBUG_LOG << iter << ") [SINGLE] After  swap between [" << mapping.from_primitive->id()
                            << "(" << mapping.from_primitive->output_memory_ptr() << ")] and [" << mapping.to_primitive->id()
                            << "(" << mapping.to_primitive->output_memory_ptr() << ")]" << std::endl;
            }
        }
    }
    return event_vec;
}

std::vector<event::ptr> loop_inst::preprocess_memory_for_body_network(int64_t current_iteration_idx) {
    std::vector<event::ptr> events;
    // Copy & Set sliced input memory
    for (size_t i = 0; i < concatenated_input_mem_mappings.size(); ++i) {
        const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
        memory::ptr mem = concatenated_input->get_sliced_mem(current_iteration_idx);
        OPENVINO_ASSERT(mem != nullptr, id(), " sliced input memory of loop is not allocated properly");
        concatenated_input->get_sliced_data_prim()->set_output_memory(mem);
    }

    // Set backedges and output memory
    for (auto& backedge_memory_mapping : backedge_memory_mappings) {
        auto event_vec = handle_buffers_for_next_iteration(backedge_memory_mapping, body_network, current_iteration_idx);
        for (auto ev : event_vec) {
            events.push_back(ev);
        }
    }

    if (!is_dynamic()) {
        // Set sliced output memory for static shape model
        // because body network generate output memory during the body network execution in dynamic model
        for (const auto& concat_output_mem_mapping : concatenated_output_mem_mappings) {
            concat_output_mem_mapping->setup_sliced_output_memory(current_iteration_idx);
        }
    }
    return events;
}

std::vector<event::ptr> loop_inst::postprocess_memory_for_body_network(int64_t current_iteration_idx) {
    std::vector<event::ptr> events;
    for (const auto& concat_output_mem_mapping : concatenated_output_mem_mappings) {
        auto sliced_data_prim = concat_output_mem_mapping->get_sliced_data_prim();
        auto output_mem_ptr = sliced_data_prim->output_memory_ptr();

        auto sliced_id = sliced_data_prim->id();
        if (body_network->has_event(sliced_id)) {
            auto ev = body_network->get_primitive_event(sliced_id);
            if (ev) ev->wait();
        }
        memory::ptr new_sliced_mem = concat_output_mem_mapping->get_or_create_sliced_mem(current_iteration_idx,
                                                                                    output_mem_ptr->get_layout());
        auto ev = new_sliced_mem->copy_from(body_network->get_stream(), *output_mem_ptr);
        if (ev) {
            events.push_back(ev);
        }
    }
    return events;
}
}  // namespace cldnn
