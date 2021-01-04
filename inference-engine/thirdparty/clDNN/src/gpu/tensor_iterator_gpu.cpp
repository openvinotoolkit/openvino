/*
// Copyright (c) 2020 Intel Corporation
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

#include "tensor_iterator_inst.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "math_utils.h"
#include "register_gpu.hpp"
#include "mutable_data_inst.h"
#include "input_layout_inst.h"
#include <vector>

namespace cldnn {
namespace gpu {
struct tensor_iterator_gpu : typed_primitive_impl<tensor_iterator> {
    const tensor_iterator_node& outer;
    using port_map_collection = tensor_iterator::port_map_collection;

    explicit tensor_iterator_gpu(const tensor_iterator_node& outer) : outer(outer) {}

    struct backedge_memory_binding {
        memory_impl* from_mem;
        memory_impl* to_mem;
        memory_impl::ptr backup;
        primitive_id from_id;
        primitive_id to_id;
        bool is_optimized;
    };

    struct input_memory_binding {
        memory_impl* from_mem;
        memory_impl* to_mem;
        int iteration_elements = 0;
        int offset = 0;
    };

    // helpers
    static size_t get_datatype_size(const memory_impl* mem) {
        if (mem->get_layout().data_type == data_types::f16)
            return 2;
        return 4;
    }

    static memory_impl* get_output_memory(network_impl::ptr body_network, primitive_id prim_id) {
        return &body_network->get_primitive(prim_id).get()->output_memory();
    }

    static bool check_if_can_be_optimized(backedge_memory_binding backedge, const tensor_iterator_inst& instance) {
        if (instance.get_body_network()->get_primitive(backedge.from_id)->dependencies().size() == 1)
            return false;
        auto node_feeding_backedge = instance.get_body_network()->get_primitive(backedge.from_id)->dependencies()[0];
        for (auto node_dep : node_feeding_backedge->dependencies()) {
            void* ptr1 = node_dep->output_memory().lock();
            void* ptr2 = backedge.to_mem->lock();
            if (ptr1 == ptr2) {
                return false;
            }
            node_dep->output_memory().unlock();
            backedge.to_mem->unlock();
        }
        return true;
    }

    static void copy_memory(primitive_id instance_id, memory_impl* from, memory_impl* to, size_t elements_to_copy, size_t source_offset = 0, size_t destination_offset = 0) {
        if (from->get_layout().data_type != to->get_layout().data_type)
            CLDNN_ERROR_MESSAGE(instance_id,"incompatible datatypes");
        size_t bytes_per_element = get_datatype_size(from);
        mem_lock<uint8_t> from_lock{ *from };
        mem_lock<uint8_t> to_lock{ *to };
        std::copy(from_lock.begin() + (source_offset * bytes_per_element),
                  from_lock.begin() + (source_offset * bytes_per_element) + (elements_to_copy * bytes_per_element),
                  to_lock.begin() + (destination_offset * bytes_per_element));
    }

    static void copy_entire_buffer(primitive_id instance_id, memory_impl* from, memory_impl* to, size_t destination_offset = 0) {
        copy_memory(instance_id, from, to, from->get_layout().get_linear_size(), 0, destination_offset);
    }

    // memory pools
    std::set<memory_impl::ptr> croped_mem_pool;
    std::map<primitive_id, memory_impl::ptr> croped_input_mem_pool;

    void process_internal_memory(tensor_iterator_inst& instance,
                                 std::vector<backedge_memory_binding>& backedge_mem,
                                 std::vector<input_memory_binding>& iteration_mem) {
        auto body_network = instance.get_body_network();
        const auto& ports_desc = instance.get_port_desc();
        
        for (int memory_num = 0; memory_num < instance.inputs_memory_count(); memory_num++) {
            memory_impl& memory = instance.input_memory(memory_num);
            auto input_desc = tensor_iterator_node::find_input_port_description(memory_num, ports_desc);
            if (input_desc == nullptr)
                CLDNN_ERROR_MESSAGE(instance.id(), "tensor iterator routing info incomplete");

            // handle memory
            if (input_desc->axis >= 0) { // checks if it's a memory to iterate through
                layout croped_layout
                    = instance.get_body_network()->get_primitive(input_desc->to)->output_memory().get_layout();
                memory_impl::ptr croped_mem = instance.get_network().get_engine().allocate_memory(croped_layout, 0);
                croped_input_mem_pool[input_desc->to] = croped_mem;
                int lin_size = static_cast<int>(croped_layout.get_linear_size());
                input_memory_binding memory_binding;
                memory_binding.from_mem = &memory;
                memory_binding.to_mem = croped_mem.get();
                memory_binding.iteration_elements = lin_size;
                iteration_mem.push_back(memory_binding);
                body_network->set_input_data(input_desc->to, *croped_mem.get());
            }
            else { // "normal" mem
                if (memory.get_layout().data_type != body_network->get_primitive(input_desc->to)->output_memory().get_layout().data_type)
                    CLDNN_ERROR_MESSAGE(instance.id(), "incompatible datatypes");

                body_network->set_input_data(input_desc->to, memory);
            }

            // checking if memory is a destination of a backedge
            for (auto back_edge_desc : ports_desc.back_edges) { //todo: what if node is both input & output?
                if (memory_num == back_edge_desc.to) {
                    //find corresponding input of the backedge
                    for (auto output : body_network->get_outputs()) {
                        if (output->id() == back_edge_desc.from + instance.node.backedge_suffix) {
                            backedge_memory_binding mem_bind;
                            mem_bind.from_mem = get_output_memory(body_network, output->id());
                            mem_bind.to_mem = get_output_memory(body_network, input_desc->to);
                            mem_bind.from_id = output->id();
                            mem_bind.to_id = input_desc->to;
                            mem_bind.is_optimized = false;
                            mem_bind.backup = instance.get_network().get_engine().allocate_memory(mem_bind.to_mem->get_layout(), 0);

                            if (mem_bind.to_mem->get_layout().data_type != mem_bind.to_mem->get_layout().data_type)
                                CLDNN_ERROR_MESSAGE(instance.id(), "incompatible datatypes");
                            copy_entire_buffer(instance.id(), mem_bind.to_mem, mem_bind.from_mem);
                            copy_entire_buffer(instance.id(), mem_bind.to_mem, mem_bind.backup.get());
                            backedge_mem.push_back(mem_bind);
                            break;
                        }
                    }
                }
            }
        }
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, tensor_iterator_inst& instance) override {
        auto ev = instance.get_network().get_engine().create_user_event(instance.get_network().get_stream_id(), false);
        for (auto& e : events)
            e->wait();

        //shortcut refs
        auto body_network = instance.get_body_network();
        const auto& ports_desc = instance.get_port_desc();

        //manage internal memory
        std::vector<backedge_memory_binding> backedge_mem;
        std::vector<input_memory_binding> iteration_mem;
        process_internal_memory(instance, backedge_mem, iteration_mem);

        //find an output memory
        std::string main_internal_output_id = ports_desc.output_ports.at(0);
        if (instance.node.is_output_working_as_backedge())
            main_internal_output_id += instance.node.backedge_suffix;

        auto body_output_mem_result = std::find_if(body_network->get_outputs().begin(),
                                                   body_network->get_outputs().end(),
                                                   [&](auto output) {return output->id() == main_internal_output_id;});

        if (body_output_mem_result == body_network->get_outputs().end())
            CLDNN_ERROR_MESSAGE(instance.id(), "incompatible datatypes");

        memory_impl* body_output_mem = &(*body_output_mem_result)->output_memory();
        size_t output_mem_offset = 0;
        size_t ti_output_mem_iter_size = body_output_mem->get_layout().get_linear_size();

        // memory read-write optimization
        // it makes output nodes write directly to input memory
        const bool enable_memory_rw_opt = true;
        if (enable_memory_rw_opt) {
            for (auto& backedge : backedge_mem) {
                if (check_if_can_be_optimized(backedge, instance)) {
                    body_network->set_input_data(backedge.to_id, *backedge.from_mem);
                    backedge.is_optimized = true;
                }
            }
        }

        // execute internal topology
        for (int i = 0; i < instance.node.iterations; i++) {
           // copy input mem
            for (auto& iter_mem : iteration_mem) {
                copy_memory(instance.id(), iter_mem.from_mem, iter_mem.to_mem, iter_mem.iteration_elements, iter_mem.offset);
                iter_mem.offset += iter_mem.iteration_elements;
            }
            body_network->execute(events);
            //copy output
            copy_entire_buffer(instance.id(), body_output_mem, &instance.output_memory(), output_mem_offset);
            output_mem_offset += ti_output_mem_iter_size;

            // copy back_edges
            for (auto edge_mem_bind : backedge_mem) {
                if (!edge_mem_bind.is_optimized) {
                    copy_entire_buffer(instance.id(), edge_mem_bind.from_mem, edge_mem_bind.to_mem);
                }
            }
        }
        
        //restore previous inputs' state
        for (auto edge_mem_bind : backedge_mem) {
            if (edge_mem_bind.is_optimized) {
                body_network->set_input_data(edge_mem_bind.to_id, *edge_mem_bind.to_mem);
            }
            else {
                copy_entire_buffer(instance.id(), edge_mem_bind.backup.get(), edge_mem_bind.to_mem);
            }
        }
        dynamic_cast<cldnn::user_event*>(ev.get())->set();
        return ev;
    }

    static primitive_impl* create(const tensor_iterator_node& arg) { return new tensor_iterator_gpu(arg); }
};

namespace detail {
attach_tensor_iterator_gpu::attach_tensor_iterator_gpu() {
    implementation_map<tensor_iterator>::add(
                                std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                tensor_iterator_gpu::create);
    implementation_map<tensor_iterator>::add(
                                std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                tensor_iterator_gpu::create);
}
}  // namespace detail

}  // namespace gpu
}  // namespace cldnn
