// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct slice : public primitive_base<slice> {
    CLDNN_DECLARE_PRIMITIVE(slice)

    slice() : primitive_base("", {}) {}

    /// @brief Constructs slice primitive.
    /// @param id This primitive id.
    /// @param inputs List of primitive ids.
    slice(const primitive_id& id,
          const std::vector<input_info>& inputs,
          bool is_start_constant = false,
          bool is_step_constant = false,
          bool is_axes_constant = false)
        : primitive_base{id, inputs},
          is_start_constant(is_start_constant),
          is_step_constant(is_step_constant),
          is_axes_constant(is_axes_constant) {}

    bool is_start_constant = false;
    bool is_step_constant = false;
    bool is_axes_constant = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, is_start_constant);
        seed = hash_combine(seed, is_step_constant);
        seed = hash_combine(seed, is_axes_constant);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }

        const auto& rhs_slice = downcast<const slice>(rhs);
        return is_start_constant == rhs_slice.is_start_constant && is_step_constant == rhs_slice.is_step_constant &&
               is_axes_constant == rhs_slice.is_axes_constant;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<slice>::save(ob);
        ob << is_start_constant;
        ob << is_step_constant;
        ob << is_axes_constant;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<slice>::load(ib);
        ib >> is_start_constant;
        ib >> is_step_constant;
        ib >> is_axes_constant;
    }
};
}  // namespace cldnn
