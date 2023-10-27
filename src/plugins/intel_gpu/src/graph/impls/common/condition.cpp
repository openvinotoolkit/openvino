// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "condition_inst.h"
#include "implementation_map.hpp"
#include "register.hpp"

#include <algorithm>
#include <vector>

namespace cldnn {
namespace common {

struct condition_impl : typed_primitive_impl<condition> {
    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::common::condition_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<condition_impl>(*this);
    }

    explicit condition_impl(const condition_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<condition>());
        const auto& node = arg.as<condition>();
        _node_id = node.id();
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, condition_inst& instance) override {
        for (auto& a : events) {
            a->wait();
        }
        auto ev = instance.get_network().get_stream().create_user_event(false);
        set_node_params(instance.get_node());

        auto pred = condition_inst::get_pred_from_memory(instance.pred_memory_ptr(), instance.get_network().get_stream());
        network::ptr executed_net = pred? instance.get_net_true() : instance.get_net_false();
        auto branch = pred? instance.get_branch_true() : instance.get_branch_false();
        GPU_DEBUG_LOG << "predicate: " << (pred ? "True" : "False") << std::endl;

        // Set input memory of inner network before its execution
        for (size_t mem_idx = 0; mem_idx < instance.inputs_memory_count(); mem_idx++) {
            const primitive_id& input_external_id = instance.dependencies().at(mem_idx).first->id();
            auto iter = branch.input_map.find(input_external_id);
            if (iter != branch.input_map.end()) {
                const primitive_id& input_internal_id = iter->second;
                auto mem_ptr = instance.input_memory_ptr(mem_idx);
                executed_net->set_input_data(input_internal_id, mem_ptr);
                GPU_DEBUG_LOG << "Inner net - Inputs[" << mem_idx << "]" << mem_ptr->get_layout().to_short_string() << std::endl;
            }
        }

        executed_net->execute({});

        // Update output layout of impl_param in condition_inst
        instance.update_output_layout();

        // Set output memory of condition_inst to inner network output memory after inner network execution
        instance.postprocess_output_memory(executed_net, branch);

        ev->set();
        return ev;
    }

    static std::unique_ptr<primitive_impl> create(const condition_node& arg, const kernel_impl_params&) {
        return make_unique<condition_impl>(arg);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

private:
    primitive_id _node_id;
};

namespace detail {

attach_condition_common::attach_condition_common() {
    implementation_map<condition>::add(impl_types::common,
                                    shape_types::dynamic_shape,
                                    condition_impl::create,
                                    {},
                                    {});
    implementation_map<condition>::add(impl_types::common, condition_impl::create, {});
}

}  // namespace detail
}  // namespace common
}  // namespace cldnn

// TODO: Change code like cldnn::loop
ASSIGN_TYPE_NAME(cldnn::common::condition_impl)
