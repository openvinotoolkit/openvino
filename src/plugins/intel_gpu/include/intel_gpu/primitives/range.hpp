// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {
struct range: public primitive_base<range> {
    CLDNN_DECLARE_PRIMITIVE(range)

    range() : primitive_base("", {}) {}

    /// @brief Constructs range primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitive id vector.
    /// @param output_layout requested range output layout
    range(const primitive_id& id,
          const std::vector<input_info>& inputs,
          const layout& output_layout)
        : primitive_base(id, inputs, 1, {output_layout.data_type}),
          output_layout(output_layout) {}

    range(const primitive_id& id,
          const std::vector<input_info>& inputs,
          const data_types data_type)
        : primitive_base(id, inputs, 1, {optional_data_type(data_type)}),
          output_layout({}) {}

    /// @brief requested range output layout
    layout output_layout;

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<range>::save(ob);
        ob << output_layout;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<range>::load(ib);
        ib >> output_layout;
    }
};
}  // namespace cldnn
