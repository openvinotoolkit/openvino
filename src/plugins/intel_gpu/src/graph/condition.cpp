// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "condition_inst.h"
#include "program_node.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(condition)

const size_t idx_branch_true    = 0;
const size_t idx_branch_false   = 1;

static std::map<primitive_id, layout> get_out_layout_map(cldnn::program::ptr prog) {
    std::map<primitive_id, layout> out_layout_map;
    for (auto& o : prog->get_outputs()) {
        out_layout_map.insert({o->id(), o->get_output_layout()});
    }
    return out_layout_map;
}

static std::map<primitive_id, layout> get_out_layout_map(cldnn::network::ptr net) {
    std::map<primitive_id, layout> out_layout_map;
    for (auto& o : net->get_outputs()) {
        out_layout_map.insert({o->id(), o->get_output_layout()});
    }
    return out_layout_map;
}

static std::vector<layout> get_output_layouts(std::map<primitive_id, layout>&& outputs, const std::map<size_t, cldnn::primitive_id> &io_output_map) {
    std::vector<layout> out_layouts;
    for (auto out : outputs) {
        for (auto& io_output : io_output_map) {
            auto inner_prim_id = io_output.second;
            if (out.first == inner_prim_id) {
                out_layouts.push_back(out.second);
            }
        }
    }
    OPENVINO_ASSERT(out_layouts.size() > 0, "Not found any matched output");
    return out_layouts;
}

/*
    Calc_output_layout method is called only when output layout is invalidated.
    It means, that it is called when:
    1) It has never been called.
    2) Dependency has changed output layout.
    In this both cases, we need to recalc branch_true and branch_false.
    !* We can be sure, that this method was called AT LEAST once during graph compilation.*!
*/
layout condition_inst::calc_output_layout(condition_node const& /* node */, kernel_impl_params const& impl_param) {
    OPENVINO_ASSERT(static_cast<bool>(impl_param.desc->output_data_types[0]) == false, "Output data type forcing is not supported for condition_node!");
    OPENVINO_ASSERT(impl_param.get_input_layout(0).count() == 1, "layout of compare_data of condition should be {1,1,1,1}");

    OPENVINO_ASSERT(impl_param.inner_progs.size() == 2, "If(Condition) contains incorrect number of inner programs ", impl_param.inner_progs.size());
    OPENVINO_ASSERT(impl_param.io_output_maps.size() == 2, "If(Condition) contains incorrect number of io output maps ", impl_param.io_output_maps.size());

    auto layouts_true  = get_output_layouts(get_out_layout_map(impl_param.inner_progs[idx_branch_true]),  impl_param.io_output_maps[idx_branch_true]);
    auto layouts_false = get_output_layouts(get_out_layout_map(impl_param.inner_progs[idx_branch_false]), impl_param.io_output_maps[idx_branch_false]);

    CLDNN_ERROR_LAYOUT_MISMATCH(impl_param.desc->id,
                                "Branch true output layout",
                                layouts_true[0],
                                "branch false output layout",
                                layouts_false[0],
                                "Layout of the branches should be the same.");

    return layouts_true[0];
}

template <class T>
static bool convert_data(memory::ptr mem, stream& stream) {
    mem_lock<T, mem_lock_type::read> lock_data{mem, stream};
    return (static_cast<float>(*lock_data.data()) != 0.f);
}

bool condition_inst::get_pred_from_memory(memory::ptr mem, stream& stream) {
    auto mem_dt = mem->get_layout().data_type;
    switch (mem_dt) {
        case cldnn::data_types::f32:
            return convert_data<float>(mem, stream);
        case cldnn::data_types::f16:
            return convert_data<half_t>(mem, stream);
        case cldnn::data_types::i64:
            return convert_data<int64_t>(mem, stream);
        case cldnn::data_types::i32:
            return convert_data<int32_t>(mem, stream);
        case cldnn::data_types::i8:
            return convert_data<int8_t>(mem, stream);
        case cldnn::data_types::u8:
            return convert_data<uint8_t>(mem, stream);
        case cldnn::data_types::bin:
        default:
            return convert_data<uint32_t>(mem, stream);
    }
}

template<typename ShapeType>
std::vector<layout> condition_inst::calc_output_layouts(condition_node const& node, kernel_impl_params const& impl_param) {
    if (impl_param.inner_nets.empty()) {
        OPENVINO_ASSERT(impl_param.inner_progs.empty() == false, "The count of inner programs should not be zero");
        auto layouts_true  = get_output_layouts(get_out_layout_map(impl_param.inner_progs[idx_branch_true]),  impl_param.io_output_maps[idx_branch_true]);
        auto layouts_false = get_output_layouts(get_out_layout_map(impl_param.inner_progs[idx_branch_false]), impl_param.io_output_maps[idx_branch_false]);

        if (layouts_true[0].is_static() && layouts_true[0] == layouts_false[0]) {
            return {layouts_true[0]};
        } else {
            OPENVINO_ASSERT(layouts_true[0].get_rank() == layouts_false[0].get_rank(), "dynamic rank is not supported");
            return {layout{ov::PartialShape::dynamic(layouts_true[0].get_rank()), layouts_true[0].data_type, layouts_true[0].format }};
        }
    } else {
        auto& memory_deps = impl_param.memory_deps;
        OPENVINO_ASSERT(memory_deps.count(0) > 0, "The count of memory deps should not be zero");
        auto mem_ptr = memory_deps.at(0);
        auto pred = condition_inst::get_pred_from_memory(mem_ptr, impl_param.get_stream());
        if (pred) {
            auto layouts_true  = get_output_layouts(get_out_layout_map(impl_param.inner_nets[idx_branch_true]),  impl_param.io_output_maps[idx_branch_true]);
            return {layouts_true[0]};
        } else {
            auto layouts_false = get_output_layouts(get_out_layout_map(impl_param.inner_nets[idx_branch_false]), impl_param.io_output_maps[idx_branch_false]);
            return {layouts_false[0]};
        }
    }
}

template std::vector<layout> condition_inst::calc_output_layouts<ov::PartialShape>(condition_node const& node, const kernel_impl_params& impl_param);

std::string condition_inst::to_string(condition_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    json_composite condition_info;

    node_info->add("condition info", condition_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

/*
Condition primitive is resuing memory with the input.
*/
condition_inst::typed_primitive_inst(network& network, condition_node const& node)
    : parent(network, node),
      _net_true(network::allocate_network(node.get_program().get_engine(), node.get_branch_true().inner_program, true)),
      _net_false(network::allocate_network(node.get_program().get_engine(), node.get_branch_false().inner_program, true)) {
    this->set_inner_networks({_net_true, _net_false});
}

event::ptr condition_inst::execute(const std::vector<event::ptr>& events) {
    const auto primitive_id = id();
    OPENVINO_ASSERT(_has_valid_input, primitive_id, " has invalid/unset input");
    GPU_DEBUG_GET_INSTANCE(debug_config);

    std::vector<event::ptr> dependencies;

    on_execute();

    GPU_DEBUG_TRACE << id() << ": execute " << _impl->get_kernel_name() << std::endl;

    if (_exec_deps.empty() && dependencies.empty()) {
        dependencies = events;
    } else {
        auto queue_type = get_network().get_stream().get_queue_type();
        // Prepare dependencies events in case of OOO queue, CPU implementation,
        // or optimized_out impl which has CPU users (needs_completion_event() && !is_output() condition)
        if (queue_type == QueueTypes::out_of_order || _impl->is_cpu() || (can_be_optimized() && needs_completion_event() && !is_output())) {
            dependencies.reserve(dependencies.size() + _exec_deps.size());
            for (auto& input : _exec_deps) {
                auto id = input->id();
                try {
                    // if the requested event does not exists it means that it has not been executed, so the processing_order is
                    // wrong or synchronization failed.
                    auto ev = get_network().get_primitive_event(id);
                    dependencies.emplace_back(ev);
                } catch (const std::out_of_range& oor) {
                    OPENVINO_ASSERT(false, "[GPU] execution order corrupted: ", oor.what());
                }
            }
        }
    }

    {
        GPU_DEBUG_PROFILED_STAGE(instrumentation::pipeline_stage::inference);
        auto ev = _impl->execute(dependencies, *this);

        GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
            get_network().get_stream().wait_for_events({ev});

            if (ev != nullptr) {
                auto profiling_info = ev->get_profiling_info();
                for (const auto &interval : profiling_info) {
                    if (interval.stage == cldnn::instrumentation::profiling_stage::executing) {
                        GPU_DEBUG_CODE(stage_prof.set_custom_stage_duration(interval.value->value()));
                    }
                }
            }
        }

        if (is_dynamic()) {
            update_shape();
            reset_shape_change();
        }

        return ev;
    }
}
}  // namespace cldnn
