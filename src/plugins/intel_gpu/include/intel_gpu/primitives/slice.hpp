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

    /// @brief Bit positions of non-constant inputs in `non_constant_input_mask`.
    enum SLICE_NON_CONST_INPUT { START = 0x1, STEP = (0x1 << 1), AXES = (0x1 << 2) };

    /// @brief Constructs slice primitive.
    /// @param id This primitive id.
    /// @param inputs List of primitive ids.
    slice(const primitive_id& id,
          const std::vector<input_info>& inputs,
          int32_t non_constant_input_mask = 0,
          const std::vector<int64_t>& start_constant = {},
          const std::vector<int64_t>& step_constant = {},
          const std::vector<int64_t>& axes_constant = {})
        : primitive_base{id, inputs},
          non_constant_input_mask(non_constant_input_mask),
          start_constant(start_constant),
          step_constant(step_constant),
          axes_constant(axes_constant) {}

    // Empty constant vector is ambiguous (0-element constant); the mask resolves it.
    int32_t non_constant_input_mask = 0;
    std::vector<int64_t> start_constant;
    std::vector<int64_t> step_constant;
    std::vector<int64_t> axes_constant;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, non_constant_input_mask);
        seed = hash_range(seed, start_constant.begin(), start_constant.end());
        seed = hash_range(seed, step_constant.begin(), step_constant.end());
        seed = hash_range(seed, axes_constant.begin(), axes_constant.end());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }

        const auto& rhs_slice = downcast<const slice>(rhs);
        return non_constant_input_mask == rhs_slice.non_constant_input_mask && start_constant == rhs_slice.start_constant &&
               step_constant == rhs_slice.step_constant && axes_constant == rhs_slice.axes_constant;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<slice>::save(ob);
        ob << non_constant_input_mask;
        ob << start_constant;
        ob << step_constant;
        ob << axes_constant;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<slice>::load(ib);
        ib >> non_constant_input_mask;
        ib >> start_constant;
        ib >> step_constant;
        ib >> axes_constant;
    }
};
}  // namespace cldnn
