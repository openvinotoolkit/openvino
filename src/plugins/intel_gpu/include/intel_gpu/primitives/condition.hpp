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
/// @n   Applies comparision using pred primitive which has 1D tensor or scalar value
struct condition : public primitive_base<condition> {
    CLDNN_DECLARE_PRIMITIVE(condition)

    /// @brief branch has compiled program, input_map and output_map
    ///
    struct branch {
        std::map<primitive_id, primitive_id> input_map;
        std::map<size_t, primitive_id> output_map;
        program::ptr inner_program;

        std::string str() {
            std::stringstream ss;
            ss << "branch: {input_map : [(outer_id,inner_id),";
            for (auto& in_iter : input_map) {
                ss << "(" << in_iter.first << "," << in_iter.second << "),";
            }
            ss << "],";

            ss << " output_map : [(outer_idx,inner_id),";
            for (auto& out_iter : output_map) {
                ss << "(" << out_iter.first << ","<< out_iter.second << "),";
            }
            ss << "]}";
            return ss.str();
        }
    };

    /// @brief Constructs condition primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (pred, inputs(optional)).
    ///                           pred is condition's predicate primitive which has scalar value determining whether to execute branch_true or branch_false.
    ///                           sometimes, if
    /// @param branch_true        Branch containg primitives, which will be executed when pred is true. then body in ngraph
    /// @param branch_false       Branch containg primitives, which will be executed when pred is false. else body in ngraph
    /// @param output_padding     Optional padding for output from primitive.
    condition(const primitive_id& id,
            const std::vector<input_info>& inputs,
            const branch& branch_true,
            const branch& branch_false,
            const size_t num_outputs = 1)
        : primitive_base(id, inputs, {padding()}, {optional_data_type()}, num_outputs),
        branch_true(branch_true),
        branch_false(branch_false) {}

    branch branch_true;
    branch branch_false;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override { return {}; }
};

static inline std::ostream& operator<< (std::ostream& os, condition::branch& info) {
    os << info.str();
    return os;
}
}  // namespace cldnn
  /// @}
  /// @}
  /// @}
