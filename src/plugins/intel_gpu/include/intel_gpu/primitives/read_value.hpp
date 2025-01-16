// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/type/element_type.hpp"
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
    /// @param output_layouts Memory layouts
    read_value(const primitive_id& id,
               const std::vector<input_info>& inputs,
               const std::string& variable_id,
               const std::vector<layout>& output_layouts,
               const ov::element::Type& user_specified_type = ov::element::undefined)
            : primitive_base(id, inputs, output_layouts.size()),
              variable_id{variable_id},
              output_layouts{output_layouts},
              user_specified_type(user_specified_type) {
        for (size_t output_idx = 0; output_idx < output_layouts.size(); output_idx++) {
            output_data_types[output_idx] = optional_data_type(output_layouts[output_idx].data_type);
        }
    }

    std::string variable_id;
    std::vector<layout> output_layouts;
    ov::element::Type user_specified_type;

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const read_value>(rhs);

        return variable_id == rhs_casted.variable_id &&
               user_specified_type == rhs_casted.user_specified_type;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<read_value>::save(ob);
        ov::element::Type_t data_type = user_specified_type;
        ob << variable_id;
        ob << output_layouts.size();
        for (const auto& layout : output_layouts)
            ob << layout;
        ob << make_data(&data_type, sizeof(ov::element::Type_t));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<read_value>::load(ib);
        ov::element::Type_t data_type = ov::element::Type_t::undefined;
        ib >> variable_id;
        size_t output_layouts_size;
        ib >> output_layouts_size;
        output_layouts.resize(output_layouts_size);
        for (size_t i = 0; i < output_layouts_size; i++) {
            ib >> output_layouts[i];
        }
        ib >> make_data(&data_type, sizeof(ov::element::Type_t));
        user_specified_type = data_type;
    }
};
}  // namespace cldnn
