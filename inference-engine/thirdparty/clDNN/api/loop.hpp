// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>
#include <functional>
#include "primitive.hpp"
#include "topology.hpp"

#define DEFAULT_MAX_NUM_ITERATION 256
namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{
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

    struct io_primitive_map {
        /// @brief Constructs a mapping from external input/output primitive to input/output primitive in body topology
        ///
        /// @param external_id Primitive id of input of loop or output of body network.
        /// @param internal_id Primitive id of input of body network.
        /// @param axis Axis to iterate through. Negative value means the axis will not iterate through and start, end, stride arguments will be ignored.
        /// @param start Index where the iteration starts from. Applies only when axis >=0.
        /// @param end Index where iteration ends. Negative value means counting indexes from the end. Applies only when axis >=0.
        /// @param stride Step of iteration. Negative value means backward iteration. Applies only when axis >=0.
        io_primitive_map(primitive_id external_id, primitive_id internal_id,
            int64_t axis = -1, int64_t start = 0, int64_t end = -1, int64_t stride = 1) :
            external_id(external_id),
            internal_id(internal_id),
            axis(axis),
            start(start),
            end(end),
            stride(stride)
            {}
        primitive_id external_id;
        primitive_id internal_id;
        int64_t axis;
        int64_t start;
        int64_t end;
        int64_t stride;
    };

    struct backedge_mapping {
        /// @brief Constructs a mapping from output of body topology to input of body topology for the next iteration
        ///
        /// @param from Output data primitive id of body topology
        /// @param to Input data primitive id of body topology
        backedge_mapping(primitive_id from, primitive_id to)
            : from(from), to(to) {}
        primitive_id from;
        primitive_id to;
    };

    /// @brief Constructs loop primitive.
    ///
    /// @param id This primitive id.
    /// @param inputs Input data primitive ids.
    /// @param body Topology to be recurrently executed.
    /// @param trip_count_id Data primitive id in external topology specifying maximum number of iterations.
    ///                      Its data primitive should have 1 integer element. Negative value means infinite
    ///                      number of iteration.
    /// @param initial_condition_id Data primitive id in external topology specifying initial execution
    ///                                       condition. Its data primitive should have 1 integer element. Zero means
    ///                                       loop will not be executed, otherwise loop will be executed.
    /// @param num_iteration_id mutable_data primitive id to get the actual number of loop iterations.
    /// @param current_iteration_id Optional data primitive id in the body network to specify current iteration.
    ///                             If current_iteration_id is specified but body does not have data whose primitive
    ///                             id is same as current_iteration_id, data primitive will be added in the body network.
    /// @param condition_id Optional data primitive id in the body network to specify execution condition
    ///                               for the next iteration. Its data primitive should have 1 integer element. Zero means
    ///                               loop will not be executed, otherwise loop will be executed.  If condition_id
    ///                               is specified but body does not have data whose primitive id is same as condition_id,
    ///                               data primitive will be added in the body network.
    /// @param primitive_map Rules to map input of loop or output of body topology to input of the body topology
    /// @param back_edges Output data primitive id.
    /// @param output_padding     Optional padding for output from primitive.
    loop(const primitive_id& id,
        const std::vector<primitive_id>& inputs,
        const topology& body,
        const primitive_id& trip_count_id,
        const primitive_id& initial_condition_id,
        const primitive_id& num_iteration_id,
        const std::vector<io_primitive_map>& input_primitive_maps,
        const std::vector<io_primitive_map>& output_primitive_maps,
        const std::vector<backedge_mapping>& back_edges,
        int64_t max_iteration = -1,
        const primitive_id& current_iteration_id = primitive_id(),
        const primitive_id& condition_id = primitive_id(),
        const padding& output_padding = padding())
            : primitive_base(id, inputs, output_padding),
              body(body),
              trip_count_id(trip_count_id),
              initial_execution_id(initial_condition_id),
              num_iteration_id(num_iteration_id),
              current_iteration_id(current_iteration_id),
              condition_id(condition_id),
              input_primitive_maps(input_primitive_maps),
              output_primitive_maps(output_primitive_maps),
              back_edges(back_edges),
              max_iteration(max_iteration)
              {}

    /// @brief Topology to be recurrently executed.
    topology body;

    /// @brief Data primitive id in external topology specifying maximum number of iterations.
    primitive_id trip_count_id;

    /// @brief Data primitive id in external topology specifying initial execution condition.
    primitive_id initial_execution_id;

    /// @brief mutable_data primitive id to get the actual number of loop iterations.
    primitive_id num_iteration_id;

    /// @brief Data primitive id in the body network to store current iteration
    primitive_id current_iteration_id;

    /// @brief Data primitive id in the body network to store execution condition
    primitive_id condition_id;

    /// @brief Rules to map input or output data of loop layer onto input or output data of body topology.
    std::vector<io_primitive_map> input_primitive_maps;
    std::vector<io_primitive_map> output_primitive_maps;

    /// @brief Rules to transfer data from body outputs at one iteration to body input at the next iteration.
    std::vector<backedge_mapping> back_edges;

    int64_t max_iteration;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret{
            std::ref(trip_count_id), std::ref(initial_execution_id), std::ref(num_iteration_id)
        };
        // add external_id in dependencies if not exist
        for (const auto& mapping : input_primitive_maps) {
            auto target = std::find(input.begin(), input.end(), mapping.external_id);
            if (target == input.end()) {
                ret.push_back(std::ref(mapping.external_id));
            }
        }
        return ret;
    }
};

/// @}
/// @}
/// @}
}  // namespace cldnn
