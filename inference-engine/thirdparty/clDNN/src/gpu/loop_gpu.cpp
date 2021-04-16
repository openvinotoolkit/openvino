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
#include <vector>
#include <algorithm>

namespace cldnn {
namespace gpu {
struct loop_gpu : typed_primitive_impl<loop> {
    const loop_node& node;

    explicit loop_gpu(const loop_node& node) : node(node) {}

    struct backedge_memory_binding {
        memory_impl* from_mem;
        memory_impl* to_mem;
        memory_impl::ptr backup;
        primitive_id from_id;
        primitive_id to_id;
        bool is_optimized;
    };

    struct external_memory_binding {
        primitive_id id;
        memory_impl::ptr from_mem;
        memory_impl::ptr to_mem;
        int iteration_elements = 0;
        int offset = 0;
    };

    // helpers
    static bool check_if_can_be_optimized(backedge_memory_binding& backedge, const loop_inst& instance) {
        // cannot optimized if input memory == output memory
        const auto& output_prim = instance.get_body_network()->get_primitive(backedge.from_id);
        if (output_prim->dependencies().size() == 1) {
            return false;
        }
        // can optimize if backedge.to_mem.data != backedge.from_mem.data
        const auto& node_feeding_backedge = output_prim->dependencies().front();
        for (auto node_dep : node_feeding_backedge->dependencies()) {
            mem_lock<void> output_mem{ node_dep->output_memory() };
            mem_lock<void> input_mem{ *backedge.to_mem };
            if (output_mem.data() == input_mem.data()) {
                return false;
            }
        }
        return true;
    }

    // // memory pools
    std::set<memory_impl::ptr> croped_mem_pool;
    std::map<primitive_id, memory_impl::ptr> croped_input_mem_pool;

    static memory_impl::ptr get_outer_output_memory(loop_inst& instance, const primitive_id& external_id) {
        if (external_id == instance.id()) {
            memory_impl& memory = instance.output_memory();
            return (memory_impl::ptr) &memory;
        } else {
            const auto outputPrim = instance.get_network().get_primitive(external_id);
            if (!outputPrim) {
                CLDNN_ERROR_MESSAGE(instance.id(), "loop primitive_map is incomplete");
            }
            memory_impl& memory = outputPrim->output_memory();
            return (memory_impl::ptr)&memory;
        }
    }

    static void set_primitive_memory(network_impl& network, const primitive_id& id, memory_impl& memory) {
        std::shared_ptr<cldnn::primitive_inst> prim = network.get_primitive(id);
        prim->set_output_memory(memory);
    }

    void set_output_memory(loop_inst& instance, std::vector<external_memory_binding>& output_iteration_mem) {
        //  - if axis = -1,
        //      from_mem = to_mem
        //  - if axis >= 0
        //      from_mem != to_mem
        //      iteration_elements(stride), offset => should be set
        auto body_network = instance.get_body_network();
        const auto& output_primitive_map = node.get_output_primitive_map();
        for (const auto& primitive_map : output_primitive_map) {
            const primitive_id& external_id = primitive_map.external_id;
            const primitive_id& internal_id = primitive_map.internal_id;
            if (primitive_map.axis < 0) {
                memory_impl::ptr memory = get_outer_output_memory(instance, external_id);
                set_primitive_memory(*body_network, internal_id, *memory);
            } else {
                memory_impl::ptr to_mem = get_outer_output_memory(instance, external_id);
                auto output_prim = body_network->get_primitive(internal_id);
                layout croped_layout = output_prim->output_memory().get_layout();
                memory_impl::ptr croped_mem = instance.get_network().get_engine().allocate_memory(croped_layout, 0);
                croped_input_mem_pool[internal_id] = croped_mem;
                const int linear_size = static_cast<int>(croped_layout.get_linear_size());
                set_primitive_memory(*body_network, internal_id, *croped_mem);

                external_memory_binding memory_binding;
                memory_binding.id = primitive_map.internal_id;
                memory_binding.from_mem = croped_mem;
                memory_binding.to_mem = to_mem;
                memory_binding.iteration_elements = linear_size;
                output_iteration_mem.push_back(memory_binding);
            }
        }
    }

    void set_input_memory(loop_inst& instance,
                          std::vector<backedge_memory_binding>& backedge_mem,
                          std::vector<external_memory_binding>& iteration_mem) {
        auto body_network = instance.get_body_network();
        const auto& input_primitive_map = node.get_input_primitive_map();

        const int inputs_memory_count = instance.inputs_memory_count();
        for (int memory_num = 0; memory_num < inputs_memory_count; memory_num++) {
            const primitive_id& input_external_id = instance.dependencies().at(memory_num)->id();
            if (input_external_id == node.get_trip_count_id() ||
                input_external_id == node.get_initial_execution_id()) {
                continue;
            }
            memory_impl& memory = instance.input_memory(memory_num);
            auto input_pm_ptrs = node.find_primitive_mappings(input_external_id, input_primitive_map);
            if (input_pm_ptrs.size() == 0) {
                CLDNN_ERROR_MESSAGE(instance.id(), "loop primitive_map is incomplete");
            }
            for (const loop::primitive_mapping * input_pm_ptr : input_pm_ptrs) {
                const auto& input_pm = *input_pm_ptr;

                // handle memory
                if (input_pm.axis >= 0) { // checks if it's a memory to iterate through
                    layout croped_layout
                        = instance.get_body_network()->get_primitive(input_pm.internal_id)->output_memory().get_layout();
                    memory_impl::ptr croped_mem = instance.get_network().get_engine().allocate_memory(croped_layout, 0);
                    croped_input_mem_pool[input_pm.internal_id] = croped_mem;
                    const int linear_size = static_cast<int>(croped_layout.get_linear_size());
                    external_memory_binding memory_binding;
                    memory_binding.id = input_pm.internal_id;
                    memory_binding.from_mem = (memory_impl::ptr)&memory;
                    memory_binding.to_mem = croped_mem;
                    memory_binding.iteration_elements = linear_size;
                    iteration_mem.push_back(memory_binding);
                    body_network->set_input_data(input_pm.internal_id, *croped_mem.get());
                } else { // "normal" mem
                    if (memory.get_layout().data_type != body_network->get_primitive(input_pm.internal_id)->output_memory().get_layout().data_type) {
                        CLDNN_ERROR_MESSAGE(instance.id(), "incompatible datatypes");
                    }
                    body_network->set_input_data(input_pm.internal_id, memory);
                }

                // checking if memory is a destination of a backedge
                const auto& back_edges = node.get_back_edges();
                for (const auto& back_edge : back_edges) { //todo: what if node is both input & output?
                    if (input_pm.internal_id != back_edge.to) {
                        continue;
                    }
                    //find corresponding input of the backedge
                    for (const auto& body_output : body_network->get_outputs()) {
                        if (body_output->id() != back_edge.from) {
                            continue;
                        }
                        backedge_memory_binding mem_bind;
                        mem_bind.from_mem = &body_network->get_primitive(back_edge.from)->output_memory();
                        mem_bind.to_mem = &body_network->get_primitive(back_edge.to)->output_memory();
                        mem_bind.from_id = back_edge.from;
                        mem_bind.to_id = back_edge.to;
                        mem_bind.is_optimized = false;
                        mem_bind.backup = instance.get_network().get_engine().allocate_memory(mem_bind.to_mem->get_layout(), 0);
                        if (mem_bind.to_mem->get_layout().data_type != mem_bind.to_mem->get_layout().data_type) {
                            CLDNN_ERROR_MESSAGE(instance.id(), "incompatible datatypes");
                        }

                        copy_entire_buffer(*mem_bind.to_mem, *mem_bind.from_mem);
                        // copy_entire_buffer(*mem_bind.to_mem, *mem_bind.backup.get());
                        backedge_mem.push_back(mem_bind);
                        break;
                    }
                }
            }
        }
    }

    // extract int from data primitive
    int64_t read_int(memory_impl& mem) {
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

    static void copy_buffer(const primitive_id& src_id, cldnn::network_impl& src_net,
                            const primitive_id& dst_id, cldnn::network_impl& dst_net,
                            const size_t size, const size_t src_offset = 0, const size_t dst_offset = 0) {
        // TODO(cldnn loop): if not used, this should be removed
        std::shared_ptr<cldnn::primitive_inst> src_data = src_net.get_primitive(src_id);
        std::shared_ptr<cldnn::primitive_inst> dst_data = dst_net.get_primitive(dst_id);
        assert(src_data->type() == cldnn::data::type_id() || src_data->type() == cldnn::mutable_data::type_id());
        assert(dst_data->type() == cldnn::data::type_id() || dst_data->type() == cldnn::mutable_data::type_id());

        memory_impl& src_mem = src_data->output_memory();
        memory_impl& dst_mem = dst_data->output_memory();
        copy_buffer(src_mem, dst_mem, size, src_offset, dst_offset);
    }

    static void copy_entire_buffer(memory_impl& src_mem, memory_impl& dst_mem, size_t destination_offset = 0) {
        copy_buffer(src_mem, dst_mem, src_mem.get_layout().get_linear_size(), 0, destination_offset);
    }

    static void write_int(memory_impl& mem, int64_t input) {
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
        int64_t trip_count = read_int(trip_count_mem);
        if (trip_count < 0) {
            trip_count = std::numeric_limits<int64_t>::max(); // infinity loop
        }

        // read initial execution condition from outer network
        const primitive_id& initial_execution_id = node.get_initial_execution_id();
        memory_impl& initial_execution_mem = outer_network.get_primitive(initial_execution_id)->output_memory();
        int64_t execution_condition = read_int(initial_execution_mem);

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

        std::vector<backedge_memory_binding> backedge_mem;
        std::vector<external_memory_binding> input_iteration_mem;
        std::vector<external_memory_binding> output_iteration_mem;

        // TODO: move out of execute_impl
        // output memory must be set before input_memory to set backedge memory properly
        set_output_memory(instance, output_iteration_mem);
        set_input_memory(instance, backedge_mem, input_iteration_mem);

        // TODO: move out of execute_impl
        // memory read-write optimization
        // it makes output nodes write directly to input memory
        const bool enable_memory_rw_opt = true;
        if (enable_memory_rw_opt) {
            for (auto& backedge : backedge_mem) {
                // check input buffer != output buffer
                if (!check_if_can_be_optimized(backedge, instance)) {
                    continue;
                }
                body_network->set_input_data(backedge.to_id, *backedge.from_mem);
                backedge.is_optimized = true;
            }
        }

        int64_t current_iteration = 0;
        if (node.is_current_iteration_used()) {
            write_int(*current_iteration_mem, current_iteration);
        }

        const auto input_ids = body_network->get_input_ids();
        const auto output_ids = body_network->get_output_ids();
        while (current_iteration < trip_count && execution_condition) {
            // copy input mem
            for (auto& iter_mem : input_iteration_mem) {
                copy_buffer(*iter_mem.from_mem, *iter_mem.to_mem, iter_mem.iteration_elements, iter_mem.offset);
                iter_mem.offset += iter_mem.iteration_elements;
            }

            body_network->execute(events);
            body_network->reset_execution();

            // copy input mem
            for (auto& iter_mem : output_iteration_mem) {
                copy_buffer(*iter_mem.from_mem, *iter_mem.to_mem, iter_mem.iteration_elements, 0, iter_mem.offset);
                iter_mem.offset += iter_mem.iteration_elements;
            }

            // update index & execution condition for the next iteration
            if (node.is_current_iteration_used()) {
                current_iteration = read_int(*current_iteration_mem);
                ++current_iteration;
                write_int(*current_iteration_mem, current_iteration);
            } else {
                ++current_iteration;
            }
            if (node.is_execution_condition_used()) {
                execution_condition = read_int(*execution_condition_mem);
            }

            // copy back_edges
            for (auto edge_mem_bind : backedge_mem) {
                if (!edge_mem_bind.is_optimized) {
                    copy_entire_buffer(*edge_mem_bind.from_mem, *edge_mem_bind.to_mem);
                }
            }
        }

        const primitive_id& num_iteration_id = node.get_num_iteration_id();
        memory_impl& num_iteration_mem = outer_network.get_primitive(num_iteration_id)->output_memory();
        write_int(num_iteration_mem, current_iteration);

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
