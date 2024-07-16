// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief Returns shape of input primitive.
struct shape_of : public primitive_base<shape_of> {
    CLDNN_DECLARE_PRIMITIVE(shape_of)

    shape_of() : primitive_base("", {}) {}

    /// @brief Constructs shape_of primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_data_type type of output values. can be i32 and i64.
    shape_of(const primitive_id& id,
             const input_info& input,
             const data_types output_data_type)
        : primitive_base(id, {input}, 1, {optional_data_type{output_data_type}}) {}

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<shape_of>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<shape_of>::load(ib);
    }
};
}  // namespace cldnn
