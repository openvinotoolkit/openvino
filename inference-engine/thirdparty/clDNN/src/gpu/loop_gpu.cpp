// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "loop_inst.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "math_utils.h"
#include "register_gpu.hpp"
#include "mutable_data_inst.h"
#include "input_layout_inst.h"
#include "memory_impl.h"
#include <vector>
#include <algorithm>

namespace cldnn {
namespace gpu {
struct loop_gpu : typed_primitive_impl<loop> {
    const loop_node& node;

    explicit loop_gpu(const loop_node& node) : node(node) {}

    // read scala value from data primitive
    static int64_t read_scalar_value(memory_impl& mem) {
        int64_t trip_count = 0;
        const layout& prim_layout = mem.get_layout();

        switch (prim_layout.data_type) {
        case data_types::u8: {
            mem_lock<uint8_t> lock_prim_output{mem};
            trip_count = *lock_prim_output.data();
            break;
        }
        case data_types::i8: {
            mem_lock<int8_t> lock_prim_output{mem};
            trip_count = *lock_prim_output.data();
            break;
        }
        case data_types::i32: {
            mem_lock<int32_t> lock_prim_output{mem};
            trip_count = *lock_prim_output.data();
            break;
        }
        case data_types::i64: {
            mem_lock<int64_t> lock_prim_output{mem};
            trip_count = *lock_prim_output.data();
            break;
        }
        default:
            assert(false);
        }
        return trip_count;
    }

    static void write_scalar_value(memory_impl& mem, int64_t input) {
        const layout& prim_layout = mem.get_layout();

        switch (prim_layout.data_type) {
        case data_types::u8: {
            assert(input >= std::numeric_limits<uint8_t>::min() &&
                   input <= std::numeric_limits<uint8_t>::max());
            mem_lock<uint8_t> lock_prim_output{mem};
            *lock_prim_output.data() = static_cast<uint8_t>(input);
            break;
        }
        case data_types::i8: {
            assert(input >= std::numeric_limits<int8_t>::min() &&
                   input <= std::numeric_limits<int8_t>::max());
            mem_lock<int8_t> lock_prim_output{mem};
            *lock_prim_output.data() = static_cast<int8_t>(input);
            break;
        }
        case data_types::i32: {
            assert(input >= std::numeric_limits<int32_t>::min() &&
                   input <= std::numeric_limits<int32_t>::max());
            mem_lock<int32_t> lock_prim_output{mem};
            *lock_prim_output.data() = static_cast<int32_t>(input);
            break;
        }
        case data_types::i64: {
            mem_lock<int64_t> lock_prim_output{mem};
            *lock_prim_output.data() = input;
            break;
        }
        default:
            assert(false);
        }
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, loop_inst& instance) override {
        auto& outer_network = instance.get_network();
        const uint32_t& net_id = instance.get_network().get_id();
        auto ev = outer_network.get_engine().create_user_event(net_id, false);

        auto body_network = instance.get_body_network();

        if (!instance.preproc_memories_done) {
            instance.preprocess_output_memory();
            instance.preprocess_input_memory();
            instance.preprocess_backedge_memory();
            instance.preproc_memories_done = true;
        }

        // read trip_count from outer network
        const primitive_id& trip_count_id = node.get_trip_count_id();
        memory_impl& trip_count_mem = outer_network.get_primitive(trip_count_id)->output_memory();
        int64_t trip_count = read_scalar_value(trip_count_mem);
        if (trip_count < 0) {
            const int64_t max_iteration = node.get_max_iteration();
            trip_count = max_iteration;
        }

        // read initial execution condition from outer network
        const primitive_id& initial_execution_id = node.get_initial_execution_id();
        memory_impl& initial_execution_mem = outer_network.get_primitive(initial_execution_id)->output_memory();
        int64_t execution_condition = read_scalar_value(initial_execution_mem);

        // shortcut of current_iteration memory in body network (slice of input)
        memory_impl* current_iteration_mem = nullptr;
        if (node.is_current_iteration_used()) {
            const primitive_id& current_iteration_id = node.get_current_iteration_id();
            current_iteration_mem = &body_network->get_primitive(current_iteration_id)->output_memory();
        }

        // shortcut of execution_condition memory in body network
        memory_impl* execution_condition_mem = nullptr;
        if (node.is_execution_condition_used()) {
            const primitive_id& condition_id = node.get_condition_id();
            execution_condition_mem = &body_network->get_primitive(condition_id)->output_memory();
        }

        int64_t current_iteration = 0;
        if (node.is_current_iteration_used()) {
            write_scalar_value(*current_iteration_mem, current_iteration);
        }

        const auto& concatenated_input_mem_mappings = instance.concatenated_input_mem_mappings;
        const auto& concatenated_output_mem_mappings = instance.concatenated_output_mem_mappings;

        // Set sliced input data
        for (size_t i = 0; i < concatenated_input_mem_mappings.size(); ++i) {
            const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
            memory_impl::ptr mem = concatenated_input.get_sliced_mem(0);
            if (mem) {
                body_network->set_input_data(concatenated_input.sliced_data_prim->id(), *mem);
            } else {
                CLDNN_ERROR_MESSAGE(node.id(), "sliced input memory of loop is not allocated properly");
            }
        }

        std::vector<event_impl::ptr> loop_carried_dep(events.begin(), events.end());

        while (current_iteration < trip_count && execution_condition) {
            // Copy & Set sliced input memory
            for (size_t i = 0; i < concatenated_input_mem_mappings.size(); ++i) {
                const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
                memory_impl::ptr mem = concatenated_input.get_sliced_mem(current_iteration);
                if (mem) {
                    concatenated_input.sliced_data_prim->set_output_memory(*mem);
                } else {
                    CLDNN_ERROR_MESSAGE(node.id(), "sliced input memory of loop is not allocated properly");
                }
            }

            // Set backedges
            for (const auto& backedge_memory_mapping : instance.backedge_memory_mappings) {
                backedge_memory_mapping.setup_iteration(current_iteration);
            }

            // Set sliced output memory
            for (const auto& concat_output_mem_mapping : concatenated_output_mem_mappings) {
                concat_output_mem_mapping.setup_concatenated_output_memory(current_iteration);
            }

            // execute body network
            body_network->execute(loop_carried_dep);

            loop_carried_dep.clear();
            for (const auto& backedge : node.get_back_edges()) {
                event_impl::ptr body_event = body_network->get_primitive_event(backedge.from);
                loop_carried_dep.emplace_back(body_event);
            }

            //TODO: "curreint_iteration primitive and execution_condition is prepared
            //as they are presented in the ngraph opset document for loop operation.
            //However they are not being used yet and only TensorIterator which has fixed sequence length is being validated.
            if (node.is_current_iteration_used()) {
                write_scalar_value(*current_iteration_mem, current_iteration);
            }
            if (node.is_execution_condition_used()) {
                execution_condition = read_scalar_value(*execution_condition_mem);
            }
            // update index & execution condition for the next iteration
            ++current_iteration;
        }

        body_network->reset_execution();

        // Concatenate sliced output to the outer network
        for (size_t i = 0; i < concatenated_output_mem_mappings.size(); ++i) {
            const auto& concat_output = concatenated_output_mem_mappings.at(i);
            concat_output.restore_concatenated_mem();
        }

        const primitive_id& num_iteration_id = node.get_num_iteration_id();
        memory_impl& num_actual_iterations_mem = outer_network.get_primitive(num_iteration_id)->output_memory();
        write_scalar_value(num_actual_iterations_mem, current_iteration);

        dynamic_cast<cldnn::user_event*>(ev.get())->set();
        return ev;
    }

    static primitive_impl* create(const loop_node& arg) { return new loop_gpu(arg); }
};

namespace detail {
attach_loop_gpu::attach_loop_gpu() {
    implementation_map<loop>::add({{engine_types::ocl, loop_gpu::create}});
}
}  // namespace detail

}  // namespace gpu
}  // namespace cldnn
