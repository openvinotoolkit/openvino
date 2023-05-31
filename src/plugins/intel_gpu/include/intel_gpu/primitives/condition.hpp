// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"
#include <vector>

namespace cldnn {

/// @brief Adds primitive, which works like "if".
///
/// @details
/// @n   Applies comparision between 2 inputs.
/// @n   Compare data - sizes of that input specifes the range of the comparison.
/// @n   Offset - offset in memory, when comparing values.
struct condition : public primitive_base<condition> {
    CLDNN_DECLARE_PRIMITIVE(condition)

    /// @brief
    struct branch_info {
        std::map<primitive_id, primitive_id> input_map;
        std::map<size_t, primitive_id> output_map;
        // topology::ptr topology_ptr;
        program::ptr inner_program;

        std::string str() {
            std::stringstream ss;
            ss << "branch_info: { " << std::endl;
            ss<< "* input_map : [(outer_id,inner_id),";
            for (auto& in_iter : input_map) {
                ss << "(" << in_iter.first << "," << in_iter.second << "),";
            }
            ss << "]," << std::endl;

            ss << "* output_map : [(outer_idx,inner_id),";
            for (auto& out_iter : output_map) {
                ss << "(" << out_iter.first << ","<< out_iter.second << "),";
            }
            ss << "]" << std::endl;
            ss << "}" << std::endl;
            return ss.str();
        }
    };

    /// @brief Constructs condition primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param input              An identifier of primitive which is an input for newly created
    ///                           condition primitive.
    /// @param branch_true        Topology containg primitives, which will be executed when comparsion results is true
    ///                           true.
    /// @param branch_false       Topology containg primitives, which will be executed when comparsion results
    ///                           false..
    /// @param compare_Data       An identifier of primitive which contains compare values
    /// @param func               Used function during comparison.
    /// @param offset             Offset for compare data.
    /// @param output_padding     Optional padding for output from primitive.
    condition(const primitive_id& id,
            const std::vector<input_info>& inputs,
            const branch_info& branch_true,
            const branch_info& branch_false,
            const padding& output_padding = padding())
        : primitive_base(id, inputs, {output_padding}),
        branch_true(branch_true),
        branch_false(branch_false) {}

    /// @brief An identifier of topology, which will be executed when comparison returns true.
    topology topology_true;
    /// @brief An identifier of topology, which will be executed when comparison returns false.
    topology topology_false;
    /// @brief An identifier of primitive which contains compare values.
    primitive_id compare_data;

    branch_info branch_true;
    branch_info branch_false;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override { return {}; }
};

static inline std::ostream& operator<< (std::ostream& os, condition::branch_info& info) {
    os << info.str();
    return os;
}
}  // namespace cldnn
  /// @}
  /// @}
  /// @}
