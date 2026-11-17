// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "condition_inst.h"
#include "data_inst.h"
#include "registry/implementation_map.hpp"
#include "register.hpp"

#include <algorithm>
#include <vector>
namespace cldnn {
namespace common {

struct condition_impl : typed_primitive_impl<condition> {
    using parent = typed_primitive_impl<condition>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::common::condition_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<condition_impl>(*this);
    }

    condition_impl() : parent() {}

    explicit condition_impl(const condition_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<condition>());
        const auto& node = arg.as<condition>();
        _node_id = node.id();
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, condition_inst& instance) override {
        // Wait for condition statement event only, and pass all other events to sub-network directly
        if (!events.empty()) {
            events[0]->wait();
        }

        auto& stream = instance.get_network().get_stream();
        set_node_params(instance.get_node());

        auto pred = condition_inst::get_pred_from_memory(instance.pred_memory_ptr(), stream);
        network::ptr executed_net = pred ? instance.get_net_true() : instance.get_net_false();
        auto branch = pred ? instance.get_branch_true() : instance.get_branch_false();
        bool can_skip_subgraph = branch.inner_program->can_be_optimized();
        if (!can_skip_subgraph) {
            executed_net->set_shape_predictor(instance.get_network().get_shape_predictor());
        }

        GPU_DEBUG_LOG << "predicate: " << (pred ? "True" : "False") << std::endl;
        GPU_DEBUG_LOG << "can_skip_subgraph: " << (can_skip_subgraph ? "True" : "False") << std::endl;

        std::vector<event::ptr> output_events;
        if (can_skip_subgraph) {
            for (const auto& output_mapping : branch.output_map) {
                const auto out_idx = output_mapping.first;
                const auto& output_id = output_mapping.second;
                auto output_it = std::find_if(branch.inner_program->get_outputs().begin(),
                                              branch.inner_program->get_outputs().end(),
                                              [&](const program_node* output) { return output->id() == output_id; });
                OPENVINO_ASSERT(output_it != branch.inner_program->get_outputs().end(),
                                "[GPU] Condition primitive ", instance.id(), " has no body output ", output_id,
                                " mapped to output index ", out_idx);
                const auto& output_internal_node = **output_it;
                layout output_layout = output_internal_node.get_output_layout();
                cldnn::memory::ptr output_mem_ptr = nullptr;
                if (output_internal_node.is_type<data>()) {
                    GPU_DEBUG_LOG << "- body output[" << out_idx << "] is constant" << std::endl;
                    output_mem_ptr = output_internal_node.as<data>().get_attached_memory_ptr();
                } else {
                    const auto& input_internal_node = output_internal_node.get_dependency(0);
                    if (input_internal_node.is_type<data>()) {
                        GPU_DEBUG_LOG << "- body output[" << out_idx << "] is from constant" << std::endl;
                        output_mem_ptr = input_internal_node.as<data>().get_attached_memory_ptr();
                    } else {
                        auto input_mapping =
                            std::find_if(branch.input_map.begin(), branch.input_map.end(), [&](const auto& m) {
                                return input_internal_node.id() == m.second.internal_id;
                            });
                        OPENVINO_ASSERT(input_mapping != branch.input_map.end(),
                                        "[GPU] No input of condition primitive ", instance.id(),
                                        " is mapped to body parameter ", input_internal_node.id());
                        const size_t dep_idx = input_mapping->first;
                        GPU_DEBUG_LOG << "- body output[" << out_idx
                                      << "] is from parameter (internal: " << input_internal_node.id()
                                      << ", dependency index: " << dep_idx << ")" << std::endl;
                        const auto& dep_info = instance.dependencies().at(dep_idx);
                        if (events.size() > dep_idx) {
                            output_events.push_back(events[dep_idx]);
                        }
                        output_mem_ptr = instance.input_memory_ptr(dep_idx);
                        output_layout = dep_info.first->get_output_layout(dep_info.second);
                    }
                }

                if (output_layout.get_partial_shape().rank().get_length() == 0) {
                    auto other_branch = !pred ? instance.get_branch_true() : instance.get_branch_false();
                    auto other_mapping = other_branch.output_map.find(out_idx);
                    OPENVINO_ASSERT(other_mapping != other_branch.output_map.end(),
                                    "[GPU] Condition primitive ", instance.id(),
                                    " has no output mapping for branch output index ", out_idx);
                    auto other_output = std::find_if(other_branch.inner_program->get_outputs().begin(),
                                                     other_branch.inner_program->get_outputs().end(),
                                                     [&](const program_node* output) {
                                                         return output->id() == other_mapping->second;
                                                     });
                    OPENVINO_ASSERT(other_output != other_branch.inner_program->get_outputs().end(),
                                    "[GPU] Condition primitive ", instance.id(), " has no body output ",
                                    other_mapping->second, " mapped to output index ", out_idx);
                    auto other_layout = (*other_output)->get_output_layout();
                    output_layout = condition_inst::adjust_scalar_to_1d_layout(output_layout, other_layout);
                    OPENVINO_ASSERT(output_mem_ptr != nullptr, "output_mem_ptr should not be null.");
                    GPU_DEBUG_LOG << "    output layout is updated to " << output_layout.to_short_string() << std::endl;
                }
                if (output_mem_ptr && output_mem_ptr->get_layout() != output_layout) {
                    output_mem_ptr = instance.get_network().get_engine().reinterpret_buffer(*output_mem_ptr, output_layout);
                }
                GPU_DEBUG_LOG << "    set output layout : " << output_layout.to_short_string() << std::endl;
                instance.set_output_layout(output_layout, out_idx);
                instance.set_output_memory(output_mem_ptr, out_idx != 0u, out_idx);
                instance.set_flag(ExecutionFlags::MEMORY_CHANGED);
            }
            return stream.group_events(output_events);
        }  // Set input memory of inner network before its execution
            for (size_t mem_idx = 0; mem_idx < instance.inputs_memory_count(); mem_idx++) {
                const auto& dep_info = instance.dependencies().at(mem_idx);
                const primitive_id& dep_base_id = dep_info.first->id();
                primitive_id lookup_key = dep_base_id;
                if (dep_info.second != 0)
                    lookup_key += ".out" + std::to_string(dep_info.second);

                auto mapping = branch.input_map.find(mem_idx);
                if (mapping == branch.input_map.end())
                    continue;
                const primitive_id& input_internal_id = mapping->second.internal_id;

                auto mem_ptr = instance.input_memory_ptr(mem_idx);
                const auto& dep = instance.dependencies()[mem_idx];
                auto layout = dep.first->get_impl_params()->get_output_layout(dep.second);
                if (mem_ptr) {
                    GPU_DEBUG_LOG << "Reshape input from " << mem_ptr->get_layout().to_short_string() << " to "
                                  << layout.to_short_string() << std::endl;
                    // Preallocation logic may allocate more memory than actually produced on current iteration, so
                    // we need to adjust input buffers layout
                    mem_ptr = instance.get_network().get_engine().reinterpret_buffer(*mem_ptr, layout);
                } else if (layout.count() == 0) {
                    // Use dummy memory for empty tensor
                    mem_ptr = std::make_shared<simple_attached_memory>(layout, nullptr);
                }
                OPENVINO_ASSERT(mem_ptr != nullptr, "[GPU] Can't assign nullptr memory buffer for condition primitive with id=", instance.id(), " ("
                                                    "mem_idx=", mem_idx, ", "
                                                    "external_id=", lookup_key, ", "
                                                    "internal_id=", input_internal_id, ")");
                executed_net->set_input_data(input_internal_id, mem_ptr);
                GPU_DEBUG_LOG << "Inner net - Inputs[" << mem_idx << "]: layout=" << mem_ptr->get_layout().to_short_string() << ", "
                              << "allocation_type=" << mem_ptr->get_allocation_type() << std::endl;
            }

        auto sub_net_results = executed_net->execute(events);
        // Update output layout of impl_param in condition_inst
        instance.update_output_layout();

        // Set output memory of condition_inst to inner network output memory after inner network execution
        instance.postprocess_output_memory(executed_net, branch);

        for (auto& output : sub_net_results) {
            if (output.second.get_event() != nullptr) {
                output_events.push_back(output.second.get_event());
            }
        }

        return stream.group_events(output_events);
    }

    static std::unique_ptr<primitive_impl> create(const condition_node& arg, const kernel_impl_params&) {
        return std::make_unique<condition_impl>(arg);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << _node_id;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> _node_id;
    }

private:
    primitive_id _node_id;
};

namespace detail {

attach_condition_common::attach_condition_common() {
    implementation_map<condition>::add(impl_types::common,
                                    shape_types::dynamic_shape,
                                    condition_impl::create,
                                    std::vector<data_types>{},
                                    {});
    implementation_map<condition>::add(impl_types::common, condition_impl::create, {});
}

}  // namespace detail
}  // namespace common
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::common::condition_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::condition)
