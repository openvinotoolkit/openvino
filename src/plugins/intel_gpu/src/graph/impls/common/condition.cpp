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
    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<condition_impl>(*this);
    }

    explicit condition_impl(const condition_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<condition>());
        const auto& node = arg.as<condition>();
        _node_id = node.id();
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, condition_inst& instance) override {
        for (auto& a : events) {
            a->wait();
        }
        auto ev = instance.get_network().get_stream().create_user_event(false);
        set_node_params(instance.get_node());

        bool exec_branch = choose_branch_to_exec(instance);
        memory::ptr memory_to_copy;
        if (exec_branch)
            memory_to_copy = execute_branch(instance.get_net_true(), instance.result_id(), instance.input_memory_ptr());
        else
            memory_to_copy = execute_branch(instance.get_net_false(), instance.result_id(), instance.input_memory_ptr());
        // just copy memory
        mem_lock<float, mem_lock_type::read> inp_ptr{memory_to_copy, instance.get_network().get_stream()};
        mem_lock<float, mem_lock_type::write> out_ptr{instance.output_memory_ptr(), instance.get_network().get_stream()};
        std::copy(inp_ptr.begin(), inp_ptr.end(), out_ptr.begin());
        ev->set();
        return ev;
    }

    static std::unique_ptr<primitive_impl> create(const condition_node& arg, const kernel_impl_params&) {
        return make_unique<condition_impl>(arg);
    }

    void init_kernels(const kernels_cache&) override {}

private:
    primitive_id _node_id;

    /*
    Add functions here.
    */
    bool check_condition(const float value_1, const float value_2, const cond_functions& func) const {
        switch (func) {
            case cond_functions::EQUAL:
                return value_1 == value_2;
                break;
            case cond_functions::GREATER:
                return value_1 > value_2;
                break;
            case cond_functions::LESS:
                return value_1 < value_2;
                break;
            default:
                throw("Unknown comparision function for: " + _node_id);
                break;
        }
    }

    /*
    Loop over memory and check condition.
    Returns boolean flag, which says what branch should be executed.
    */
    bool choose_branch_to_exec(condition_inst& instance) const {
        mem_lock<float, mem_lock_type::read> lock_compare_data{instance.compare_memory_ptr(), instance.get_network().get_stream()};
        auto compare_layout = instance.compare_memory().get_layout();
        auto compare_ptr = lock_compare_data.begin();

        mem_lock<float, mem_lock_type::read> lock_input{instance.input_memory_ptr(), instance.get_network().get_stream()};
        auto input_layout = instance.input_memory().get_layout();
        auto input_ptr = lock_input.begin();

        auto function = instance.argument->function;
        auto& offset = instance.argument->offset;

        for (auto b = 0; b < compare_layout.batch(); b++) {
            for (auto f = 0; f < compare_layout.feature(); f++) {
                for (auto z = 0; z < compare_layout.spatial(2); z++) {
                    for (auto y = 0; y < compare_layout.spatial(1); y++) {
                        for (auto x = 0; x < compare_layout.spatial(0); x++) {
                            tensor input_tensor{
                                batch(b + offset.batch[0]),
                                feature(f + offset.feature[0]),
                                spatial(x + offset.spatial[0], y + offset.spatial[1], z + offset.spatial[2], 0) };
                            auto input_idx = input_layout.get_linear_offset(input_tensor);
                            tensor compare_tensor{ batch(b), feature(f), spatial(x, y, z, 0) };
                            auto compare_idx = compare_layout.get_linear_offset(compare_tensor);
                            if (!check_condition(input_ptr[input_idx], compare_ptr[compare_idx], function))
                                return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    memory::ptr execute_branch(network::ptr branch,
                           const primitive_id& input_id,
                           memory::ptr input_memory) const {
        branch->set_input_data(input_id, input_memory);
        branch->execute({});
        return branch->get_outputs().at(0)->output_memory_ptr();
    }
};

namespace detail {

attach_condition_common::attach_condition_common() {
    implementation_map<condition>::add(impl_types::common, condition_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f32, format::yxfb),
    });
}

}  // namespace detail
}  // namespace common
}  // namespace cldnn

ASSIGN_TYPE_NAME(cldnn::common::condition_impl)
