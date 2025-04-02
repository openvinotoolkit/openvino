// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_expert_inst.h"
#include "data_inst.h"
#include "registry/implementation_map.hpp"
#include "register.hpp"

#include <algorithm>
#include <vector>
namespace cldnn {
namespace common {

struct moe_expert_impl : typed_primitive_impl<moe_expert> {
    using parent = typed_primitive_impl<moe_expert>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::common::moe_expert_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<moe_expert_impl>(*this);
    }

    moe_expert_impl() : parent() {}

    explicit moe_expert_impl(const moe_expert_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<moe_expert>());
        const auto& node = arg.as<moe_expert>();
        _node_id = node.id();
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, moe_expert_inst& instance) override {
        // Wait for moe_expert statement event only, and pass all other events to sub-network directly
        auto dep_event = instance.pred_inst()->get_impl_params()->out_event;
        if (dep_event)
            dep_event->wait();

        auto& stream = instance.get_network().get_stream();
        auto& prog_node = instance.get_node();
        set_node_params(prog_node);

        auto pred = moe_expert_inst::get_pred_from_memory(instance.pred_memory_ptr(), stream, instance.get_config().expert_no);
        network::ptr executed_net = instance.get_net();
        auto branch = instance.get_branch();
        bool can_skip_subgraph = !pred;
        if (!can_skip_subgraph)
            executed_net->set_shape_predictor(instance.get_network().get_shape_predictor());

        GPU_DEBUG_LOG << "predicate: " << (pred ? "True" : "False") << std::endl;
        GPU_DEBUG_LOG << "can_skip_subgraph: " << (can_skip_subgraph ? "True" : "False") << std::endl;

        std::vector<event::ptr> output_events;
        if (can_skip_subgraph) {
            auto output_mem_ptr = instance.input_memory_ptr(0);
            auto output_layout = instance.dependencies()[0].first->get_output_layout();
            GPU_DEBUG_LOG << "    set output layout : " << output_layout.to_short_string() << std::endl;
            instance.set_output_layout(output_layout, 0);
            instance.set_output_memory(output_mem_ptr, 0);
            instance.set_flag(ExecutionFlags::MEMORY_CHANGED);

            return stream.group_events(output_events);
        } else {
            // Set input memory of inner network before its execution
            for (size_t mem_idx = 0; mem_idx < instance.inputs_memory_count(); mem_idx++) {
                const primitive_id& input_external_id = instance.dependencies().at(mem_idx).first->id();
                auto iter = branch.input_map.find(input_external_id);
                if (iter != branch.input_map.end()) {
                    const primitive_id& input_internal_id = iter->second;
                    auto mem_ptr = instance.input_memory_ptr(mem_idx);
                    auto dep = instance.dependencies()[mem_idx];
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
                    OPENVINO_ASSERT(mem_ptr != nullptr, "[GPU] Can't assign nullptr memory buffer for moe_expert primitive with id=", instance.id(), " ("
                                                        "mem_idx=", mem_idx, ", "
                                                        "external_id=", input_external_id, ", "
                                                        "internal_id=", input_internal_id, ")");
                    executed_net->set_input_data(input_internal_id, mem_ptr);
                    GPU_DEBUG_LOG << "Inner net - Inputs[" << mem_idx << "]: layout=" << mem_ptr->get_layout().to_short_string() << ", "
                                  << "allocation_type=" << mem_ptr->get_allocation_type() << std::endl;
                }
            }
        }

        auto sub_net_results = executed_net->execute(events);
        // Update output layout of impl_param in moe_expert_inst
        instance.update_output_layout();

        // Set output memory of moe_expert_inst to inner network output memory after inner network execution
        instance.postprocess_output_memory(executed_net, branch);

        for (auto& output : sub_net_results)
            if (output.second.get_event() != nullptr)
                output_events.push_back(output.second.get_event());

        return stream.group_events(output_events);
    }

    static std::unique_ptr<primitive_impl> create(const moe_expert_node& arg, const kernel_impl_params&) {
        return std::make_unique<moe_expert_impl>(arg);
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

attach_moe_expert_common::attach_moe_expert_common() {
    implementation_map<moe_expert>::add(impl_types::common,
                                    shape_types::dynamic_shape,
                                    moe_expert_impl::create,
                                    std::vector<data_types>{},
                                    {});
    implementation_map<moe_expert>::add(impl_types::common, moe_expert_impl::create, {});
}

}  // namespace detail
}  // namespace common
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::common::moe_expert_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_expert)
