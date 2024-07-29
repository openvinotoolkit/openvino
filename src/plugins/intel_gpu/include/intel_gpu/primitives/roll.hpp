// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "primitive.hpp"

namespace cldnn {

/// @brief Roll-7 primitive.
struct roll : primitive_base<roll> {
    CLDNN_DECLARE_PRIMITIVE(roll)

    roll() : primitive_base("", {}) {}

    /// @brief Constructs roll primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param shift Tensor which specifies the number of places by which the elements are shifted.
    roll(const primitive_id& id,
         const input_info& input,
         const tensor& shift)
        : primitive_base(id, {input}),
          shift(shift) {}

    /// @brief Constructs roll primitive for dynamic shape.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param raw_shift raw shift vector
    /// @param raw_axes raw axes vector
    roll(const primitive_id& id,
         const input_info& input,
         const std::vector<int32_t>& raw_shift,
         const std::vector<int32_t>& raw_axes)
        : primitive_base(id, {input}),
          raw_shift(raw_shift), raw_axes(raw_axes) {}

    /// @brief Tensor which specifies the number of places by which the elements are shifted.
    tensor shift;

    /// @brief Raw shift/axes vector to calculate normalized shift when input shape becomes static
    std::vector<int32_t> raw_shift;
    std::vector<int32_t> raw_axes;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, shift.hash());
        seed = hash_range(seed, raw_shift.begin(), raw_shift.end());
        seed = hash_range(seed, raw_axes.begin(), raw_axes.end());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const roll>(rhs);

        return shift == rhs_casted.shift &&
               raw_shift == rhs_casted.raw_shift &&
               raw_axes == rhs_casted.raw_axes;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<roll>::save(ob);
        ob << shift;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<roll>::load(ib);
        ib >> shift;
    }
};

}  // namespace cldnn
