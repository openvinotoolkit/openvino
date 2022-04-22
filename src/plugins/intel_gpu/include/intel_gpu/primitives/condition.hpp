// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{
/// @brief Function, which will be used during comparison.
enum cond_functions : int32_t { EQUAL, GREATER, LESS };

/// @brief Adds primitive, which works like "if".
///
/// @details
/// @n   Applies comparision between 2 inputs.
/// @n   Compare data - sizes of that input specifes the range of the comparison.
/// @n   Offset - offset in memory, when comparing values.
struct condition : public primitive_base<condition> {
    CLDNN_DECLARE_PRIMITIVE(condition)

    /// @brief Constructs condition primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param input              An identifier of primitive which is an input for newly created
    ///                           condition primitive.
    /// @param topology_true      Topolgoy containg primitives, which will be executed when comparsion results
    ///                           true.
    /// @param topology_false     Topolgoy containg primitives, which will be executed when comparsion results
    ///                           false..
    /// @param compare_Data       An identifier of primitive which contains compare values
    /// @param func               Used function during comparison.
    /// @param offset             Offset for compare data.
    /// @param output_padding     Optional padding for output from primitive.
    condition(const primitive_id& id,
              const input_info& input,
              const topology& topology_true,
              const topology& topology_false,
              const primitive_id& compare_data,
              const cond_functions& func,
              const tensor& offset = {0, 0, 0, 0, 0},
              const primitive_id& ext_prim_id = "",
              const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, {output_padding}),
          topology_true(topology_true),
          topology_false(topology_false),
          compare_data(compare_data),
          function(func),
          offset(offset) {}

    /// @brief An identifier of topology, which will be executed when comparison returns true.
    topology topology_true;
    /// @brief An identifier of topology, which will be executed when comparison returns false.
    topology topology_false;
    /// @brief An identifier of primitive which contains compare values.
    primitive_id compare_data;
    /// @brief Used function during comparison.
    cond_functions function;
    /// @brief Offset for compare data.
    tensor offset;

protected:
    std::vector<std::pair<std::reference_wrapper<const primitive_id>, int>> get_dependencies() const override { return {{std::ref(compare_data), 0}}; }
};
}  // namespace cldnn
  /// @}
  /// @}
  /// @}
