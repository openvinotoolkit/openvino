// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "loop_inst.h"
#include "impls/implementation_map.hpp"
#include "register.hpp"
#include "mutable_data_inst.h"
#include "input_layout_inst.h"
#include <vector>
#include <algorithm>

namespace cldnn {
namespace common {
struct loop_impl : typed_primitive_impl<loop> {
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<loop_impl>(*this);
    }

    explicit loop_impl(const loop_node& arg) :
        _loop_node(&arg),
        _is_current_iteration_used(arg.is_current_iteration_used()),
        _current_iteration_id(arg.get_current_iteration_id()),
        _id(arg.id()),
        _trip_count_id(arg.get_trip_count_id()),
        _max_iteration(arg.get_max_iteration()),
        _initial_execution_id(arg.get_initial_execution_id()),
        _is_execution_condition_used(arg.is_execution_condition_used()),
        _condition_id(arg.get_condition_id()),
        _back_edges(arg.get_back_edges()),
        _num_iteration_id(arg.get_num_iteration_id()) {}

    static std::unique_ptr<primitive_impl> create(const loop_node& arg) { return make_unique<loop_impl>(arg); }

private:
    void init_kernels(const program&) override {}

    event::ptr execute_impl(const std::vector<event::ptr>& events, loop_inst& instance) override {
        auto& outer_network = instance.get_network();
        auto& stream = outer_network.get_stream();

        auto body_network = instance.get_body_network();

        auto ev = stream.create_user_event(false);

        const primitive_id& id = (_loop_node ? _loop_node->id() : _id);
        const bool is_current_iteration_used = (_loop_node ? _loop_node->is_current_iteration_used() : _is_current_iteration_used);
        if (!instance.preproc_memories_done) {
            instance.preprocess_output_memory();
            instance.preprocess_input_memory();
            instance.preprocess_backedge_memory();

            // set input data for current_iteration primitive if current_iteration is used
            if (is_current_iteration_used) {
                const primitive_id& current_iteration_id = (_loop_node ? _loop_node->get_current_iteration_id() : _current_iteration_id);
                auto current_iteration_prim = body_network->get_primitive(current_iteration_id);
                auto input_layout_prim = std::dynamic_pointer_cast<input_layout_inst>(current_iteration_prim);
                if (input_layout_prim == nullptr) {
                    CLDNN_ERROR_MESSAGE(id, "current_iteration primitive is not input_layout");
                }

                const auto& backedge_mapping = instance.get_current_iteration_backedge_mapping();
                input_layout_prim->set_data(backedge_mapping.initial_mem);
            }
            instance.preproc_memories_done = true;
        }

        // read trip_count from outer network
        bool update_num_iterations = false;
        const primitive_id& trip_count_id = (_loop_node ? _loop_node->get_trip_count_id() : _trip_count_id);
        memory::ptr trip_count_mem = outer_network.get_primitive(trip_count_id)->output_memory_ptr();
        int64_t trip_count = loop_node::read_scalar_value(trip_count_mem, stream);
        if (trip_count < 0) {
            trip_count = (_loop_node ? _loop_node->get_max_iteration() : _max_iteration);
            update_num_iterations = true;
        }

        // read initial execution condition from outer network
        const primitive_id& initial_execution_id = (_loop_node ? _loop_node->get_initial_execution_id() : _initial_execution_id);
        memory::ptr initial_execution_mem = outer_network.get_primitive(initial_execution_id)->output_memory_ptr();
        int64_t execution_condition = loop_node::read_scalar_value(initial_execution_mem, stream);

        // shortcut of execution_condition memory in body network
        memory::ptr execution_condition_mem = nullptr;
        const bool is_execution_condition_used = (_loop_node ? _loop_node->is_execution_condition_used() : _is_execution_condition_used);
        if (is_execution_condition_used) {
            const primitive_id& condition_id = (_loop_node ? _loop_node->get_condition_id() : _condition_id);
            execution_condition_mem = body_network->get_primitive(condition_id)->output_memory_ptr();
        }

        const auto& concatenated_input_mem_mappings = instance.concatenated_input_mem_mappings;
        const auto& concatenated_output_mem_mappings = instance.concatenated_output_mem_mappings;

        // Set sliced input data
        for (size_t i = 0; i < concatenated_input_mem_mappings.size(); ++i) {
            const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
            memory::ptr mem = concatenated_input.get_sliced_mem(0);
            if (mem) {
                body_network->set_input_data(concatenated_input.sliced_data_prim->id(), mem);
            } else {
                CLDNN_ERROR_MESSAGE(id, "sliced input memory of loop is not allocated properly");
            }
        }

        std::vector<event::ptr> loop_carried_dep(events.begin(), events.end());
        int64_t current_iteration_idx = 0;
        while (current_iteration_idx < trip_count && execution_condition) {
            // Copy & Set sliced input memory
            for (size_t i = 0; i < concatenated_input_mem_mappings.size(); ++i) {
                const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
                memory::ptr mem = concatenated_input.get_sliced_mem(current_iteration_idx);
                if (mem) {
                    concatenated_input.sliced_data_prim->set_output_memory(mem);
                } else {
                    CLDNN_ERROR_MESSAGE(id, "sliced input memory of loop is not allocated properly");
                }
            }

            // Set backedges
            for (const auto& backedge_memory_mapping : instance.backedge_memory_mappings) {
                backedge_memory_mapping.setup_iteration(current_iteration_idx);
            }

            // Set sliced output memory
            for (const auto& concat_output_mem_mapping : concatenated_output_mem_mappings) {
                concat_output_mem_mapping.setup_concatenated_output_memory(current_iteration_idx);
            }

            // execute body network
            body_network->execute(loop_carried_dep);

            loop_carried_dep.clear();
            const auto& back_edges = (_loop_node ? _loop_node->get_back_edges() : _back_edges);
            for (const auto& backedge : back_edges) {
                event::ptr body_event;
                if (body_network->has_event(backedge.from))
                    body_event = body_network->get_primitive_event(backedge.from);
                loop_carried_dep.emplace_back(body_event);
            }

            //TODO: execution_condition is prepared as they are presented in the
            //      ngraph opset document for loop operation.
            // However they are not being used yet and only TensorIterator which
            // has fixed sequence length is being validated.
            if (is_execution_condition_used) {
                execution_condition = loop_node::read_scalar_value(execution_condition_mem, stream);
            }

            // update index & execution condition for the next iteration
            ++current_iteration_idx;
        }

        body_network->reset_execution();

        // Concatenate sliced output to the outer network
        for (size_t i = 0; i < concatenated_output_mem_mappings.size(); ++i) {
            const auto& concat_output = concatenated_output_mem_mappings.at(i);
            concat_output.restore_concatenated_mem();
        }

        if (update_num_iterations) {
            // update num_iterations (actual number of iterations)
            int64_t actual_iterations = 0;
            if (is_current_iteration_used) {
                const auto& backedge_mapping = instance.get_current_iteration_backedge_mapping();
                auto current_iteration_mem = backedge_mapping.from_primitive->output_memory_ptr();
                actual_iterations = loop_node::read_scalar_value(current_iteration_mem, stream);
            } else {
                actual_iterations = current_iteration_idx;
            }

            const primitive_id& num_iteration_id = (_loop_node ? _loop_node->get_num_iteration_id() : _num_iteration_id);
            memory::ptr num_actual_iterations_mem = outer_network.get_primitive(num_iteration_id)->output_memory_ptr();
            loop_node::write_scalar_value(num_actual_iterations_mem, stream, actual_iterations);
        }

        ev->set();
        return ev;
    }

private:
    loop_node const * const _loop_node = nullptr;
    bool _is_current_iteration_used = false;
    primitive_id _current_iteration_id;
    primitive_id _id;
    primitive_id _trip_count_id;
    int64_t _max_iteration = 0;
    primitive_id _initial_execution_id;
    bool _is_execution_condition_used = false;
    primitive_id _condition_id;
    std::vector<cldnn::loop::backedge_mapping> _back_edges{};
    primitive_id _num_iteration_id;
};

namespace detail {
attach_loop_common::attach_loop_common() {
    implementation_map<loop>::add(impl_types::common, loop_impl::create, {});
}
}  // namespace detail

}  // namespace common
}  // namespace cldnn
