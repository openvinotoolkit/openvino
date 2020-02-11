// Copyright (c) 2018 Intel Corporation
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

#include "condition_inst.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "math_utils.h"
#include "register_gpu.hpp"

#include <algorithm>
#include <vector>

namespace cldnn {
namespace gpu {

struct condition_gpu : typed_primitive_impl<condition> {
    const condition_node& outer;

    explicit condition_gpu(const condition_node& outer) : outer(outer) {}

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, condition_inst& instance) override {
        for (auto& a : events) {
            a->wait();
        }
        auto ev = instance.get_network().get_engine().create_user_event(instance.get_network().get_id(), false);

        bool exec_branch = choose_branch_to_exec(instance);
        memory_impl::ptr memory_to_copy;
        if (exec_branch)
            memory_to_copy = (memory_impl::ptr) &execute_branch(instance.get_net_true(), instance.result_id(), instance.input_memory());
        else
            memory_to_copy = (memory_impl::ptr) &execute_branch(instance.get_net_false(), instance.result_id(), instance.input_memory());
        // just copy memory
        mem_lock<float> inp_ptr{memory_to_copy};
        mem_lock<float> out_ptr{instance.output_memory()};
        std::copy(inp_ptr.begin(), inp_ptr.end(), out_ptr.begin());
        dynamic_cast<cldnn::user_event*>(ev.get())->set();  // set as complete
        return ev;
    }

    static primitive_impl* create(const condition_node& arg) { return new condition_gpu(arg); }

private:
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
                throw("Unknown comparision function for: " + outer.id());
                break;
        }
    }

    /*
    Loop over memory and check condition.
    Returns boolean flag, which says what branch should be executed.
    */
    bool choose_branch_to_exec(condition_inst& instance) const {
        mem_lock<float> lock_compare_data{instance.compare_memory()};
        auto compare_layout = instance.compare_memory().get_layout();
        auto compare_ptr = lock_compare_data.begin();

        mem_lock<float> lock_input{instance.input_memory()};
        auto input_layout = instance.input_memory().get_layout();
        auto input_ptr = lock_input.begin();

        auto function = instance.argument.function;
        auto& offset = instance.argument.offset;
        auto& range = compare_layout.size;

        for (auto b = 0; b < range.batch[0]; b++) {
            for (auto f = 0; f < range.feature[0]; f++) {
                for (auto z = 0; z < range.spatial[2]; z++) {
                    for (auto y = 0; y < range.spatial[1]; y++) {
                        for (auto x = 0; x < range.spatial[0]; x++) {
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

    memory_impl& execute_branch(network_impl::ptr branch,
                                const primitive_id& input_id,
                                memory_impl& input_memory) const {
        branch->set_input_data(input_id, input_memory);
        branch->execute({});
        return branch->get_outputs().at(0)->output_memory();
    }
};

namespace detail {

attach_condition_gpu::attach_condition_gpu() {
    implementation_map<condition>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                        condition_gpu::create);
    implementation_map<condition>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb),
                                        condition_gpu::create);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
