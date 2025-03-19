// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>
#include <functional>
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"

namespace cldnn {

///
/// @brief Adds primitive which performs recurrent execution of the topology.
///
/// @details
/// @n   The body topology for recurrent execution is described in the body
/// @n   The execution of the body topology iterates through the data in the given axis.
/// @n   Note: that only loops with fixed iteration count are being validated and supported currently.
/// @n
/// @n\b Example:
/// \code{.cpp}
/// topology body(
///     data("eltwise_operand", operand_mem),
///     eltwise("eltwise", "input", "eltwise_operand", eltwise_mode::sum)
/// );
///
/// std::vector<loop::io_primitive_map> input_primitive_maps { loop::io_primitive_map("input", "input") };
/// std::vector<loop::io_primitive_map> output_primitive_maps { loop::io_primitive_map("loop", "eltwise") };
///
/// std::vector<loop::backedge_mapping> back_edges {
///     loop::backedge_mapping("eltwise", "input")
/// };
///
/// topology topology(
///     input_layout("input", input_mem.get_layout()),
///     input_layout("trip_count", trip_count_mem.get_layout()),
///     input_layout("initial_condition", initial_condition_mem.get_layout()),
///     mutable_data("num_iteration", num_iteration_mem),
///     loop("loop", {"input"}, body,
///             "trip_count", "initial_condition", "num_iteration",
///             input_primitive_maps, output_primitive_maps, back_edges)
/// );
///
/// network network(engine, topology);
/// network.set_input_data("input", input_mem);
/// network.set_input_data("trip_count", trip_count_mem);
/// network.set_input_data("initial_condition", initial_condition_mem);
/// \endcode

struct loop : public primitive_base<loop> {
    CLDNN_DECLARE_PRIMITIVE(loop)

    loop() : primitive_base("", {}),
             max_num_iterations(0) {}

    struct io_primitive_map {
        /// @brief Constructs a mapping from external input/output primitive to input/output primitive in body topology
        ///         or a mapping from output of body topology to input of body topology for the next iteration.
        /// @param external_id Primitive id of input of loop or output of body network.
        /// @param internal_id Primitive id of input of body network.
        /// @param axis Axis to iterate through. Negative value means the axis will not iterate through and start, end, stride arguments will be ignored.
        /// @param start Index where the iteration starts from. Applies only when axis >=0.
        /// @param end Index where iteration ends. Negative value means counting indexes from the end. Applies only when axis >=0.
        /// @param stride Step of iteration. Negative value means backward iteration. Applies only when axis >=0.
        io_primitive_map(primitive_id external_id, primitive_id internal_id,
            int64_t axis = -1, int64_t start = 0, int64_t end = -1, int64_t stride = 1) :
            external_id(external_id, 0),
            internal_id(internal_id, 0),
            axis(axis),
            start(start),
            end(end),
            stride(stride) {}

        /// @brief Constructs a mapping from external input/output primitive to input/output primitive in body topology
        ///         or a mapping from output of body topology to input of body topology for the next iteration.
        /// @param external_id Primitive id of input of loop or output of body network.
        /// @param internal_id Primitive id of input of body network.
        /// @param axis Axis to iterate through. Negative value means the axis will not iterate through and start, end, stride arguments will be ignored.
        /// @param start Index where the iteration starts from. Applies only when axis >=0.
        /// @param end Index where iteration ends. Negative value means counting indexes from the end. Applies only when axis >=0.
        /// @param stride Step of iteration. Negative value means backward iteration. Applies only when axis >=0.
        io_primitive_map(input_info external_id = input_info(), input_info internal_id = input_info(),
            int64_t axis = -1, int64_t start = 0, int64_t end = -1, int64_t stride = 1) :
            external_id(std::move(external_id)),
            internal_id(std::move(internal_id)),
            axis(axis),
            start(start),
            end(end),
            stride(stride) {}

        input_info external_id;
        input_info internal_id;
        int64_t axis;
        int64_t start;
        int64_t end;
        int64_t stride;

        void save(BinaryOutputBuffer& ob) const {
            ob << external_id;
            ob << internal_id;
            ob << axis;
            ob << start;
            ob << end;
            ob << stride;
        }

        void load(BinaryInputBuffer& ib) {
            ib >> external_id;
            ib >> internal_id;
            ib >> axis;
            ib >> start;
            ib >> end;
            ib >> stride;
        }

        std::string to_string() const {
            std::stringstream ss;
            ss << "io_primitive_map " << std::endl;
            ss << "* external_id    : " << external_id.to_string() << std::endl;
            ss << "* internal_id    : " << internal_id.to_string() << std::endl;
            ss << "* axis           : " << axis << std::endl;
            ss << "* start          : " << start << std::endl;
            ss << "* end            : " << end << std::endl;
            ss << "* stride         : " << stride << std::endl;
            return ss.str();
        }

        std::string to_short_string() const {
            std::stringstream ss;
            ss << "io_primitive_map[e:" << external_id.to_string();
            ss << "," << internal_id.to_string();
            ss << "," << axis;
            ss << "," << start;
            ss << "," << end;
            ss << "," << stride << "]";
            return ss.str();
        }
    };

    struct backedge_mapping {
        /// @brief Constructs a mapping from output of body topology to input of body topology for the next iteration
        ///
        /// @param from Output data primitive id of body topology
        /// @param to Input data primitive id of body topology
        backedge_mapping(primitive_id from, primitive_id to)
            : from(from), to(to) {}
        backedge_mapping() {}
        primitive_id from;
        primitive_id to;

        void save(BinaryOutputBuffer& ob) const {
            ob << from;
            ob << to;
        }

        void load(BinaryInputBuffer& ib) {
            ib >> from;
            ib >> to;
        }
    };

    /// @brief Constructs loop primitive.
    ///
    /// @param id This primitive id.
    /// @param inputs Input data primitive ids.
    /// @param body_program body program to be recurrently executed.
    /// @param trip_count_id Data primitive id in external topology specifying maximum number of iterations.
    ///                      Its data primitive should have 1 integer element. Negative value means infinite
    ///                      number of iteration.
    /// @param first_execution_condition_id Data primitive id in external topology specifying initial execution
    ///                                       condition. Its data primitive should have 1 integer element. Zero means
    ///                                       loop will not be executed, otherwise loop will be executed.
    /// @param num_iteration_id mutable_data primitive id to get the actual number of loop iterations.
    /// @param body_current_iteration_id Optional data primitive id in the body network to specify current iteration.
    ///                             If body_current_iteration_id is specified but body does not have data whose primitive
    ///                             id is same as body_current_iteration_id, data primitive will be added in the body network.
    /// @param body_execution_condition_id Optional data primitive id in the body network to specify execution condition
    ///                               for the next iteration. Its data primitive should have 1 integer element. Zero means
    ///                               loop will not be executed, otherwise loop will be executed.  If body_execution_condition_id
    ///                               is specified but body does not have data whose primitive id is same as body_execution_condition_id,
    ///                               data primitive will be added in the body network.
    /// @param primitive_map Rules to map input of loop or output of body topology to input of the body topology
    /// @param back_edges Output data primitive id.
    loop(const primitive_id& id,
         const std::vector<input_info>& inputs,
         const program::ptr body_program,
         const primitive_id& trip_count_id,
         const primitive_id& first_execution_condition_id,
         const primitive_id& num_iteration_id,
         const std::vector<io_primitive_map>& input_primitive_maps,
         const std::vector<io_primitive_map>& output_primitive_maps,
         const std::vector<backedge_mapping>& back_edges,
         int64_t max_num_iterations = -1,
         const primitive_id& body_current_iteration_id = primitive_id(),
         const primitive_id& body_execution_condition_id = primitive_id(),
         const size_t num_outputs = 1)
            : primitive_base(id, inputs, num_outputs, {optional_data_type()}),
              body_program(std::move(body_program)),
              trip_count_id(trip_count_id),
              first_execution_condition_id(first_execution_condition_id),
              num_iteration_id(num_iteration_id),
              body_current_iteration_id(body_current_iteration_id),
              body_execution_condition_id(body_execution_condition_id),
              input_primitive_maps(input_primitive_maps),
              output_primitive_maps(output_primitive_maps),
              back_edges(back_edges),
              max_num_iterations(max_num_iterations) {
        OPENVINO_ASSERT(inputs.front().pid == num_iteration_id, "first input of inputs should be num_iteration_id");
    }

    /// @brief Body program to be recurrently executed.
    program::ptr body_program;

    /// @brief Data primitive id in external topology specifying maximum number of iterations.
    primitive_id trip_count_id;

    /// @brief Data primitive id in external topology specifying initial execution condition.
    primitive_id first_execution_condition_id;

    /// @brief mutable_data primitive id to get the actual number of loop iterations.
    primitive_id num_iteration_id;

    /// @brief Data primitive id in the body network to store current iteration
    primitive_id body_current_iteration_id;

    /// @brief Data primitive id in the body network to store execution condition
    primitive_id body_execution_condition_id;

    /// @brief Rules to map input or output data of loop layer onto input or output data of body topology.
    std::vector<io_primitive_map> input_primitive_maps;
    std::vector<io_primitive_map> output_primitive_maps;

    /// @brief Rules to transfer data from body outputs at one iteration to body input at the next iteration.
    std::vector<backedge_mapping> back_edges;

    int32_t max_num_iterations;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, id);
        return seed;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<loop>::save(ob);
        body_program->save(ob);
        ob << trip_count_id;
        ob << first_execution_condition_id;
        ob << num_iteration_id;
        ob << body_current_iteration_id;
        ob << body_execution_condition_id;
        ob << input_primitive_maps;
        ob << output_primitive_maps;
        ob << back_edges;
        ob << max_num_iterations;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<loop>::load(ib);
        body_program = std::make_shared<cldnn::program>(ib.get_engine());
        body_program->load(ib);
        ib >> trip_count_id;
        ib >> first_execution_condition_id;
        ib >> num_iteration_id;
        ib >> body_current_iteration_id;
        ib >> body_execution_condition_id;
        ib >> input_primitive_maps;
        ib >> output_primitive_maps;
        ib >> back_edges;
        ib >> max_num_iterations;
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret;
        // add external_id in dependencies if not exist
        for (const auto& mapping : input_primitive_maps) {
            auto target = std::find_if(input.begin(), input.end(),
                                    [&](const input_info& info) {
                                        return info.pid == mapping.external_id.pid;});
            if (target == input.end()) {
                ret.push_back(mapping.external_id.pid);
            }
        }
        return ret;
    }
};

}  // namespace cldnn
