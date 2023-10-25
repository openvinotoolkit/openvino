// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"

namespace cldnn {

/// @brief Returns value of the variable_id variable.
struct read_value : public primitive_base<read_value> {
    CLDNN_DECLARE_PRIMITIVE(read_value)

    read_value() : primitive_base("", {}) {}

    /// @brief Constructs ReadValue primitive.
    /// @param id This primitive id
    /// @param inputs Input parameters ids
    /// @param variable_id Variable id
    /// @param output_layout Memory layout
    read_value(const primitive_id& id,
               const std::vector<input_info>& inputs,
               const std::string& variable_id,
               const layout& output_layout)
            : primitive_base(id, inputs, {padding()}, {optional_data_type{output_layout.data_type}}),
              variable_id{variable_id},
              output_layout{output_layout} {}

    std::string variable_id;
    layout output_layout;

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const read_value>(rhs);

        return variable_id == rhs_casted.variable_id;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<read_value>::save(ob);
        ob << variable_id;
        ob << output_layout;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<read_value>::load(ib);
        ib >> variable_id;
        ib >> output_layout;
    }
};
}  // namespace cldnn
