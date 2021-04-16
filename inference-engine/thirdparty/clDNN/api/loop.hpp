// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>
#include <functional>
#include "primitive.hpp"
#include "topology.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

// TODO(cldnn loop): update example code
/// @brief Adds primitive which performs recurrent execution of the topology.
///
/// @details
/// @n   The body topology for recurrent execution is described in the body
/// @n   The execution of the body topology iterates through the data in the given axis.
/// @n\b Output primitive id:
/// @n   If output is concatenated, output primitive id will be <loop primitive id>:<output primitive id>. e.g. "loop:output"
/// @n   Otherwise, the output's primitive id will be <loop primitive id>:<output primitive id>_<iteration number>.
/// @n   e.g. "loop:output_0", "loop:output_1", "loop:output_2", ...
/// @n\b Example:
/// \code{.cpp}
/// // external topology
/// primitive_id max_iteration_id = "max_iteration";
/// primitive_id initial_condition_id = "initial_condition";
/// primitive_id input_id = "input";
/// primitive_id const_id = "const";
/// primitive_id output_id = "output";
///
/// cldnn:: topology topology;
/// topology.add<cldnn::data>({ max_iteration_id, ... });
/// topology.add<cldnn::data>({ initial_condition_id, ... });
/// topology.add<cldnn::data>({ input_id, ... });
/// topology.add<cldnn::data>({ const_id, ... });
///
///
/// // set internal body network
/// primitive_id current_iteration_id = "curr_iter";
/// primitive_id loop_condition_id = "loop_condition";
/// primitive_id input_internal_id = "input_internal";
/// primitive_id const_internal_id = "const_internal";
/// primitive_id output_internal_id = "output_internal";
///
/// cldnn:: topology topology_internal;
/// // updated by loop operator
/// topology_internal.add<cldnn::data>({current_iteration_id, ...});
/// topology_internal.add<cldnn::data>({loop_condition_id, ...});
/// topology_internal.add<cldnn::data>({input_internal_id, ...});
/// topology_internal.add<cldnn::data>({const_internal_id, ...});
///
/// topology_internal.add(
///     eltwise(output_internal_id, input_internal_id, const_internal_id, eltwise::add)
/// )
///
/// std::vector<primitive_mapping> primitive_map{
///     {primitive_mapping::INPUT, input_id, input_internal_id, axis=1}, // iterating through axis 1
///     {primitive_mapping::INPUT, const_id, const_internal_id},
///     {primitive_mapping::OUTPUT, output_id, output_internal_id, axis=1}, // will be concatenated by axis 1
/// };
///
/// std::vector<backedge_mapping> backedges {
///     {output_internal_id, input_internal_id}
/// };
///
/// cldnn::loop loop(
///     "loop",
///     {input_id, const_id},
///     topology_internal,
///     max_iteration_id,
///     initial_condition_id,
///     loop_condition_id,
///     primitive_map,
///     backedges
/// );
///
/// topology.add(loop);
/// // output will be accessble by output_id
/// \endcode

struct loop : public primitive_base<loop> {
    CLDNN_DECLARE_PRIMITIVE(loop)

    /// @brief primitive_mapping type
    enum primitive_type: int32_t {
        INPUT,
        OUTPUT,
    };

    struct primitive_mapping {
        /// @brief Constructs a mapping from external input/output primitive to input/output primitive in body topology
        ///
        /// @param external_id Primitive id of input of loop or output of body network.
        /// @param internal_id Primitive id of input of body network.
        /// @param axis Axis to iterate through. Negative value means the axis will not iterate through and start, end, stride arguments will be ignored.
        /// @param start Index where the iteration starts from. Applies only when axis >=0.
        /// @param end Index where iteration ends. Negative value means counting indexes from the end. Applies only when axis >=0.
        /// @param stride Step of iteration. Negative value means backward iteration. Applies only when axis >=0.
        primitive_mapping(primitive_type type, primitive_id external_id, primitive_id internal_id,
            int32_t axis = -1, int32_t start = 0, int32_t end = -1, int32_t stride = 1) :
            type(type),
            external_id(external_id),
            internal_id(internal_id),
            axis(axis),
            start(start),
            end(end),
            stride(stride)
            {}
        primitive_type type;
        primitive_id external_id;
        primitive_id internal_id;
        int32_t axis;
        int32_t start;
        int32_t end;
        int32_t stride;
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
    /// actual max iteration = min(trip_count, (end-start)/stride)
    ///
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
        const std::vector<primitive_mapping>& primitive_map,
        const std::vector<backedge_mapping>& back_edges,
        int32_t max_iteration = -1,
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
              primitive_map(primitive_map),
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
    std::vector<primitive_mapping> primitive_map;

    /// @brief Rules to transfer data from body outputs at one iteration to body input at the next iteration.
    std::vector<backedge_mapping> back_edges;

    int32_t max_iteration;

    static const int32_t DEFAULT_MAX_ITERATION = 128;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret{
            std::ref(trip_count_id), std::ref(initial_execution_id), std::ref(num_iteration_id)
        };
        // add external_id in dependencies if not exist
        for (const auto& mapping : primitive_map) {
            if (mapping.type == loop::OUTPUT) {
                continue;
            }
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
