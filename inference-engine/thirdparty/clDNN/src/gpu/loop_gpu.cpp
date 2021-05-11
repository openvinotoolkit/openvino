/*
// Copyright (c) 2021 Intel Corporation
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
*/

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

    static memory_impl::ptr get_external_memory(const loop_inst& instance, const primitive_id& external_id) {
            const auto outputPrim = instance.get_network().get_primitive(external_id);
            memory_impl& memory = outputPrim->output_memory();
            return (memory_impl::ptr) &memory;
    }

    std::vector<memory_impl::ptr> get_sliced_mem(const primitive_id& internal_id, const loop_inst& instance) {
        const auto& concatenated_input_mem_mappings = instance.concatenated_input_mem_mappings;
        for (const auto& mem_mapping : concatenated_input_mem_mappings) {
            if (mem_mapping.sliced_data_id == internal_id) {
                return mem_mapping.sliced_mems;
            }
        }
        const auto& concatenated_output_mem_mappings = instance.concatenated_output_mem_mappings;
        for (const auto& mem_mapping : concatenated_output_mem_mappings) {
            if (mem_mapping.concat_data_id == internal_id) {
                return mem_mapping.sliced_mems;
            }
        }
        return {}; // not found
    }

    void preprocess_output_memory(loop_inst& instance) {
        auto& engine = instance.get_network().get_engine();
        auto& concatenated_output_mem_mappings = instance.concatenated_output_mem_mappings;
        const auto& body_network = instance.get_body_network();
        const auto& output_primitive_maps = node.get_output_primitive_maps();
        concatenated_output_mem_mappings.reserve(output_primitive_maps.size());
        for (size_t i = 0; i < output_primitive_maps.size(); ++i) {
            const auto& output_mapping = output_primitive_maps.at(i);
            const primitive_id& external_id = output_mapping.external_id;
            const primitive_id& internal_id = output_mapping.internal_id;
            if (output_mapping.axis < 0) {
                memory_impl::ptr memory = get_external_memory(instance, external_id);
                body_network->get_primitive(internal_id)->set_output_memory(*memory);
            } else {
                memory_impl::ptr to_mem = get_external_memory(instance, external_id);
                auto output_prim = body_network->get_primitive(internal_id);
                layout sliced_layout = output_prim->output_memory().get_layout();

                const int max_iteration = node.get_max_iteration();
                std::vector<memory_impl::ptr> sliced_mems;
                sliced_mems.reserve(max_iteration);
                for (int i=0; i < max_iteration; ++i) {
                    memory_impl::ptr sliced_mem = engine.allocate_memory(sliced_layout, 0);
                    sliced_mems.push_back(sliced_mem);
                }

                const int linear_size = static_cast<int>(sliced_layout.get_linear_size());
                const int stride = linear_size * output_mapping.stride;
                const int start = output_mapping.start < 0? node.get_max_iteration() - 1: output_mapping.start;
                const int offset = linear_size * start;
                cldnn::loop_inst::concatenated_memory_mapping memory_mapping_info(
                    output_mapping.internal_id, output_mapping.external_id,
                    to_mem, sliced_mems, linear_size, stride, offset);
                memory_mapping_info.concat_data_prim = body_network->get_primitive(internal_id);
                concatenated_output_mem_mappings.push_back(memory_mapping_info);
            }
        }
    }

    void preprocess_input_memory(loop_inst& instance) {
        auto& engine = instance.get_network().get_engine();
        auto& iteration_mem = instance.concatenated_input_mem_mappings;
        auto body_network = instance.get_body_network();
        const size_t inputs_memory_count = instance.inputs_memory_count();
        for (size_t memory_num = 0; memory_num < inputs_memory_count; memory_num++) {
            const primitive_id& input_external_id = instance.dependencies().at(memory_num)->id();
            if (input_external_id == node.get_trip_count_id() ||
                input_external_id == node.get_initial_execution_id()) {
                continue;
            }
            memory_impl& memory = instance.input_memory(memory_num);
            auto input_map_ptrs = node.find_io_primitive_maps(input_external_id, true);
            if (input_map_ptrs.size() == 0) {
                CLDNN_ERROR_MESSAGE(instance.id(), "loop primitive_map is incomplete");
            }
            for (size_t i = 0; i < input_map_ptrs.size(); ++i) {
                const auto input_map = input_map_ptrs.at(i);
                bool is_concatenated_input = (input_map->axis >= 0);
                if (is_concatenated_input) {
                    layout sliced_layout
                        = instance.get_body_network()->get_primitive(input_map->internal_id)->output_memory().get_layout();
                    const int max_iteration = node.get_max_iteration();
                    std::vector<memory_impl::ptr> sliced_mems;
                    sliced_mems.reserve(max_iteration);
                    for (int i=0; i < max_iteration; ++i) {
                        memory_impl::ptr sliced_mem = engine.allocate_memory(sliced_layout, 0);
                        sliced_mems.push_back(sliced_mem);
                    }
                    const int linear_size = static_cast<int>(sliced_layout.get_linear_size());
                    const int stride = linear_size * input_map->stride;
                    const int start = input_map->start < 0? node.get_max_iteration() - 1: input_map->start;
                    const int offset = linear_size * start;
                    loop_inst::concatenated_memory_mapping concatenated_input_mem_mapping_info(
                        input_map->external_id, input_map->internal_id,
                        (memory_impl::ptr)&memory, sliced_mems, linear_size, stride, offset);
                    concatenated_input_mem_mapping_info.sliced_data_prim = body_network->get_primitive(input_map->internal_id);
                    iteration_mem.push_back(concatenated_input_mem_mapping_info);
                } else {
                    if (memory.get_layout().data_type != body_network->get_primitive(input_map->internal_id)->output_memory().get_layout().data_type) {
                        CLDNN_ERROR_MESSAGE(instance.id(), "incompatible datatypes");
                    }
                    body_network->set_input_data(input_map->internal_id, memory);
                }
            }
        }
    }

    void preprocess_backedge_memory(loop_inst& instance) {
        const auto& body_network = instance.get_body_network();
        const auto& back_edges = node.get_back_edges();
        auto& backedge_memory_mappgins = instance.backedge_memory_mappings;
        // checking if memory is a destination of a backedge
        for (const auto& back_edge : back_edges) {
            //find corresponding input of the backedge
            const auto input_map_ptrs = node.find_io_primitive_maps(back_edge.to, false);
            assert(input_map_ptrs.size() == 1);
            const auto& input_map = input_map_ptrs.front();
            auto backedged_sliced_output_mems = get_sliced_mem(back_edge.from, instance);
            const auto backedge_to_prim = body_network->get_primitive(back_edge.to);
            const auto backedge_from_prim = body_network->get_primitive(back_edge.from);
            memory_impl::ptr initial_mem = get_external_memory(instance, input_map->external_id);
            if (backedged_sliced_output_mems.empty()) {
                // backedge output which does not need concatenation
                // input memory = output memory = loop output memory
                const auto output_mapping = node.find_io_primitive_maps(back_edge.from, false);
                memory_impl::ptr backedge_mem;
                if (output_mapping.empty()) {
                    auto output_prim = body_network->get_primitive(back_edge.from);
                    layout output_layout = output_prim->output_memory().get_layout();
                    backedge_mem = body_network->get_engine().allocate_memory(output_layout, 0);
                } else {
                    backedge_mem = get_external_memory(instance, output_mapping.front()->external_id);
                }
                body_network->set_input_data(back_edge.to, *backedge_mem);
                body_network->set_output_memory(back_edge.from, *backedge_mem);
                backedge_memory_mappgins.emplace_back(
                    backedge_from_prim, backedge_to_prim, backedge_mem, initial_mem);
            } else {
                // backedge output which needs concatenation
                backedge_memory_mappgins.emplace_back(
                    backedge_from_prim, backedge_to_prim, backedged_sliced_output_mems, initial_mem);
            }
        }
    }

    // read scala value from data primitive
    int64_t read_scalar_value(memory_impl& mem) {
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

    static void copy_buffer(cldnn::memory_impl& src_mem, cldnn::memory_impl& dst_mem,
                            const size_t size, const size_t src_offset = 0, const size_t dst_offset = 0) {
        assert(src_mem.get_layout().data_type == dst_mem.get_layout().data_type);

        size_t bytes_per_element = data_type_traits::size_of(src_mem.get_layout().data_type);
        mem_lock<uint8_t> from_lock{ src_mem };
        mem_lock<uint8_t> to_lock{ dst_mem };

        const size_t byte_size_to_copy = size * bytes_per_element;
        const auto src = from_lock.begin() + src_offset * bytes_per_element;
        const auto dst = to_lock.begin() + (dst_offset * bytes_per_element);
        std::copy(src, src + byte_size_to_copy, dst);
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
        for (auto& e : events)
            e->wait();
        auto& outer_network = instance.get_network();
        const uint32_t& net_id = instance.get_network().get_id();
        auto ev = outer_network.get_engine().create_user_event(net_id, false);

        auto body_network = instance.get_body_network();

        // read trip_count from outer network
        const primitive_id& trip_count_id = node.get_trip_count_id();
        memory_impl& trip_count_mem = outer_network.get_primitive(trip_count_id)->output_memory();
        int64_t trip_count = read_scalar_value(trip_count_mem);
        if (trip_count < 0) {
            const int max_iteration = node.get_max_iteration();
            trip_count = max_iteration; // infinity loop
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

        // output memory must be set before input_memory to set backedge memory properly
        if (!instance.preproc_memories_done) {
            preprocess_output_memory(instance);
            preprocess_input_memory(instance);
            preprocess_backedge_memory(instance);
            instance.preproc_memories_done = true;
        }

        int64_t current_iteration = 0;
        if (node.is_current_iteration_used()) {
            write_scalar_value(*current_iteration_mem, current_iteration);
        }

        const auto& concatenated_input_mem_mappings = instance.concatenated_input_mem_mappings;
        const auto& concatenated_output_mem_mappings = instance.concatenated_output_mem_mappings;
        std::vector<event_impl::ptr> loop_carried_dep;
        while (current_iteration < trip_count && execution_condition) {
            // Copy & Set sliced input memory offset
            for (size_t i = 0; i < instance.concatenated_input_mem_mappings.size(); ++i) {
                const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
                memory_impl::ptr mem = concatenated_input.get_sliced_mem(current_iteration);
                // set input mem
                if (current_iteration == 0) {
                    body_network->set_input_data(concatenated_input.sliced_data_id, *mem);
                } else {
                    concatenated_input.sliced_data_prim->set_output_memory(*mem);
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
            if (current_iteration == 0) {
                body_network->execute(events);
            } else {
                body_network->execute(loop_carried_dep);
            }
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
        memory_impl& num_iteration_mem = outer_network.get_primitive(num_iteration_id)->output_memory();
        write_scalar_value(num_iteration_mem, current_iteration);

        dynamic_cast<cldnn::user_event*>(ev.get())->set();
        return ev;
    }

    static primitive_impl* create(const loop_node& arg) { return new loop_gpu(arg); }
};

namespace detail {
attach_loop_gpu::attach_loop_gpu() {
    std::vector<data_types> loop_data_types{ data_types::bin, data_types::u8, data_types::i8, data_types::f16,
                                             data_types::f32, data_types::i32, data_types::i64};

    std::vector<format> loop_formats{ format::bfyx, format::bfzyx, format::bfwzyx };

    for (const data_types loop_data_type : loop_data_types) {
        for (const format loop_format : loop_formats) {
            implementation_map<loop>::add(
                std::make_tuple(engine_types::ocl, loop_data_type, loop_format),
                loop_gpu::create);
        }
    }
}
}  // namespace detail

}  // namespace gpu
}  // namespace cldnn
