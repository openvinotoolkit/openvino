// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "loop_inst.h"
#include "registry/implementation_map.hpp"
#include "register.hpp"
#include "mutable_data_inst.h"
#include "input_layout_inst.h"
#include <vector>
#include <algorithm>

namespace cldnn {
namespace common {

// read scala value from data primitive
static int64_t read_scalar_value(memory::ptr mem, stream& stream) {
    int64_t trip_count = 0;
    const layout& prim_layout = mem->get_layout();

    switch (prim_layout.data_type) {
    case data_types::u8: {
        mem_lock<uint8_t> lock_prim_output{mem, stream};
        trip_count = *lock_prim_output.data();
        break;
    }
    case data_types::i8: {
        mem_lock<int8_t> lock_prim_output{mem, stream};
        trip_count = *lock_prim_output.data();
        break;
    }
    case data_types::i32: {
        mem_lock<int32_t> lock_prim_output{mem, stream};
        trip_count = *lock_prim_output.data();
        break;
    }
    case data_types::i64: {
        mem_lock<int64_t> lock_prim_output{mem, stream};
        trip_count = *lock_prim_output.data();
        break;
    }
    default:
        OPENVINO_THROW("Invalid data type : ",  ov::element::Type(prim_layout.data_type).get_type_name());
    }
    return trip_count;
}

template<typename T>
static inline void validate_input_value(int64_t input) {
    OPENVINO_ASSERT((input >= std::numeric_limits<T>::min() && input <= std::numeric_limits<T>::max()),
                "Invalid data value : ", input);
}

static void write_scalar_value(memory::ptr mem, stream& stream, int64_t input) {
    const layout& prim_layout = mem->get_layout();

    switch (prim_layout.data_type) {
    case data_types::u8: {
        validate_input_value<uint8_t>(input);
        mem_lock<uint8_t> lock_prim_output{mem, stream};
        lock_prim_output[0] = static_cast<uint8_t>(input);
        break;
    }
    case data_types::i8: {
        validate_input_value<int8_t>(input);
        mem_lock<int8_t> lock_prim_output{mem, stream};
        lock_prim_output[0] = static_cast<int8_t>(input);
        break;
    }
    case data_types::i32: {
        validate_input_value<int32_t>(input);
        mem_lock<int32_t> lock_prim_output{mem, stream};
        lock_prim_output[0] = static_cast<int32_t>(input);
        break;
    }
    case data_types::i64: {
        mem_lock<int64_t> lock_prim_output{mem, stream};
        lock_prim_output[0] = input;
        break;
    }
    default:
        OPENVINO_THROW("Invalid data type : ",  ov::element::Type(prim_layout.data_type).get_type_name());
    }
}

struct loop_impl : typed_primitive_impl<loop> {
    using parent = typed_primitive_impl<loop>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::common::loop_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<loop_impl>(*this);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    loop_impl() : parent() {}

    loop_impl(const loop_impl& other) : typed_primitive_impl<loop>(other),
        _back_edges(other._back_edges) {}

    explicit loop_impl(const loop_node& node) {
        set_node_params(node);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<loop>());
        const auto& node = arg.as<loop>();
        _back_edges = node.get_back_edges();
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, loop_inst& instance) override {
        const auto& impl_params = instance.get_impl_params();
        const auto& primitive = impl_params->typed_desc<loop>();
        auto& outer_network = instance.get_network();
        auto& stream = outer_network.get_stream();

        auto body_network = instance.get_body_network();
        int64_t current_iteration_idx = 0;
        auto ev = stream.create_user_event(false);
        const auto is_dynamic = instance.is_dynamic();

        if (is_dynamic) {
            instance.update_shape();
            if (instance.get_flag(ExecutionFlags::SHAPE_CHANGED)) {
                instance.preproc_memories_done = false;
                instance.reset_memory();
            }
        }

        body_network->set_shape_predictor(outer_network.get_shape_predictor());
        OPENVINO_ASSERT(!instance.get_num_iterations_id().empty(), "loop operation should have num_iteration_id");

        // shortcut of execution_condition memory in body network
        memory::ptr body_execution_condition_mem = nullptr;
        if (!instance.get_condition_id().empty()) {
            body_execution_condition_mem = body_network->get_primitive(instance.get_condition_id())->output_memory_ptr();
        }

        // shortcut of current_iteration memory in body network
        if (!instance.get_current_iteration_id().empty()) {
            memory::ptr body_current_iteration_mem = body_network->get_primitive(instance.get_current_iteration_id())->output_memory_ptr();
            write_scalar_value(body_current_iteration_mem, body_network->get_stream(), 0);
        }

        auto num_iterations = instance.get_num_iterations();
        GPU_DEBUG_LOG << "num_iterations : " << num_iterations << std::endl;

        // read trip_count from outer network
        int64_t trip_count = -1;
        if (!instance.get_trip_count_id().empty()) {
            memory::ptr trip_count_mem = outer_network.get_primitive(instance.get_trip_count_id())->output_memory_ptr();
            trip_count = read_scalar_value(std::move(trip_count_mem), stream);
        } else {
            OPENVINO_ASSERT(!instance.get_condition_id().empty()
                            || num_iterations > 0 || primitive->max_num_iterations > 0,
                            "num_iterations should be positive when trip_count_id is not existed");
            // If trip_count_id is not existed, the original ngraph operation is TensorIterator.
            // If num_iterations is negative, it means that TensorIterator has no concat input / output memory.
            // When it has no body_exeuction_conditio_id and num_iterations and primtive->max_num_iteartion,
            // TensorIterator has no ending condition. So it cannot terminate inner body execution loop.
            trip_count = num_iterations > 0 ? num_iterations : primitive->max_num_iterations;
        }
        GPU_DEBUG_LOG << "trip_count : " << trip_count << std::endl;

        // read initial execution condition from outer network
        int64_t execution_condition = 1;
        if (!instance.get_initial_execution_id().empty()) {
            // Wait for completion of the execution_condition of outer_network
            if (outer_network.has_event(instance.get_initial_execution_id()))
                outer_network.get_primitive_event(instance.get_initial_execution_id())->wait();
            memory::ptr first_execution_condition_mem = outer_network.get_primitive(instance.get_initial_execution_id())->output_memory_ptr();
            execution_condition = read_scalar_value(first_execution_condition_mem, stream);
        }
        GPU_DEBUG_LOG << "execution_condition: " << execution_condition << std::endl;

        // When execution_condition is false or trip_count is zero, return execute_impl without any body_network execution.
        if (!execution_condition || trip_count == 0) {
            // Update num_iterations (actual number of iterations)
            memory::ptr num_actual_iterations_mem = outer_network.get_primitive(instance.get_num_iterations_id())->output_memory_ptr();
            write_scalar_value(num_actual_iterations_mem, stream, current_iteration_idx);

            instance.update_output_layout();
            ev->set();
            return ev;
        }

        if (!instance.preproc_memories_done) {
            instance.preprocess_output_memory(num_iterations);
            instance.preprocess_input_memory(num_iterations);
            instance.preprocess_backedge_memory();
            instance.preproc_memories_done = true;
        }

        const auto& concatenated_input_mem_mappings = instance.concatenated_input_mem_mappings;
        const auto& backedge_memory_mappings = instance.backedge_memory_mappings;

        // If there are concatenated_input_mem_mappings or backedge_memory_mappings we need to wait for
        // previous tasks before accessing memory in get_sliced_mem() and setup_iteration() functions
        if (!concatenated_input_mem_mappings.empty() || !backedge_memory_mappings.empty()) {
            stream.wait_for_events(events);
        }

        // Set sliced input data
        for (size_t i = 0; i < concatenated_input_mem_mappings.size(); ++i) {
            const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
            concatenated_input->slice_mem(num_iterations);
            memory::ptr mem = concatenated_input->get_sliced_mem(0);
            OPENVINO_ASSERT(mem != nullptr, instance.id(), "sliced input memory of loop is not allocated properly");
            body_network->set_input_data(concatenated_input->get_sliced_data_prim_id(), mem);
        }

        std::vector<event::ptr> all_events;
        std::vector<event::ptr> loop_carried_dep(events.begin(), events.end());
        while (((trip_count < 0) || (current_iteration_idx < trip_count)) && execution_condition) {
            auto prev_events = instance.preprocess_memory_for_body_network(current_iteration_idx);
            for (auto& ev : prev_events) {
                loop_carried_dep.push_back(ev);
            }

            // execute body network
            body_network->execute(loop_carried_dep);

            loop_carried_dep.clear();
            for (const auto& backedge : _back_edges) {
                event::ptr body_event;
                if (body_network->has_event(backedge.from)) {
                    body_event = body_network->get_primitive_event(backedge.from);
                    loop_carried_dep.emplace_back(body_event);
                }
            }

            // Collect output events for waiting for all iterations finishing
            for (auto& out : body_network->get_outputs()) {
                auto output_id = out->id();
                if (body_network->has_event(output_id)) {
                    auto output_event = body_network->get_primitive_event(output_id);
                    all_events.push_back(output_event);
                }
            }

            // Store output of sliced_data_prim to sliced mems vector
            // After execution of body network, sliced_data_prim will has output memory buffer
            // current memory buffer move to sliced_mems and new memory buffer will be allocated in sliced_data_prim
            if (is_dynamic) {
                auto post_events = instance.postprocess_memory_for_body_network(current_iteration_idx);
                for (auto& ev : post_events) {
                    loop_carried_dep.push_back(ev);
                    all_events.push_back(ev);
                }
            }

            // execution condition is the result of body network execution
            if (body_execution_condition_mem != nullptr) {
                auto execution_id = instance.get_condition_id();
                if (body_network->has_event(execution_id)) {
                    auto ev = body_network->get_primitive_event(execution_id);
                    if (ev) ev->wait();
                }
                execution_condition = read_scalar_value(body_execution_condition_mem, body_network->get_stream());
            }
            GPU_DEBUG_IF(!execution_condition) {
                GPU_DEBUG_LOG << "body_exec_condition is false at "<< current_iteration_idx << " iteration idx" << std::endl;
            }

            current_iteration_idx++;
        }

        // Reset network and wait for all collected events
        body_network->reset_execution(false);
        stream.wait_for_events(all_events);

        // Update actual num iteration
        // update num_iterations (actual number of iterations)
        memory::ptr num_actual_iterations_mem = outer_network.get_primitive(instance.get_num_iterations_id())->output_memory_ptr();
        write_scalar_value(num_actual_iterations_mem, stream, current_iteration_idx);
        GPU_DEBUG_LOG << "current_iteration_idx(" << instance.get_num_iterations_id() << ", "
                        << num_actual_iterations_mem << ")  : " << current_iteration_idx << std::endl;

        if (is_dynamic)
            instance.update_output_layout();
        instance.postprocess_output_memory(is_dynamic, current_iteration_idx);

        ev->set();
        return ev;
    }

    static std::unique_ptr<primitive_impl> create(const loop_node& arg, const kernel_impl_params&) {
        return std::make_unique<loop_impl>(arg);
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << _back_edges;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> _back_edges;
    }

private:
    std::vector<cldnn::loop::backedge_mapping> _back_edges;
};

namespace detail {
attach_loop_common::attach_loop_common() {
    implementation_map<loop>::add(impl_types::common,
                                    shape_types::dynamic_shape,
                                    loop_impl::create,
                                    std::vector<data_types>{},
                                    {});
    implementation_map<loop>::add(impl_types::common, loop_impl::create, {});
}
}  // namespace detail

}  // namespace common
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::common::loop_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::loop)
