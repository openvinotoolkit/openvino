// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {
struct cum_sum : public primitive_base<cum_sum> {
    CLDNN_DECLARE_PRIMITIVE(cum_sum)

    cum_sum() : primitive_base("", {}) {}

    /// @brief Constructs cum_sum primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param axis Scalar axis.
    /// @param exclusive If set to true then the top element is not included in sum.
    /// @param reverse If set to true will perform the sums in reverse direction.
    cum_sum(const primitive_id& id,
            const input_info& input,
            const int64_t axis = 0,
            const bool exclusive = false,
            const bool reverse = false)
        : primitive_base(id, {input}), axis(axis), exclusive(exclusive), reverse(reverse)
    {}

    /// @brief Scalar axis.
    int64_t axis = 0;
    /// @brief If set to true then the top element is not included in sum.
    bool exclusive = false;
    /// @brief If set to true will perform the sums in reverse direction.
    bool reverse = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, axis);
        seed = hash_combine(seed, exclusive);
        seed = hash_combine(seed, reverse);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const cum_sum>(rhs);

        return axis == rhs_casted.axis &&
               exclusive == rhs_casted.exclusive &&
               reverse == rhs_casted.reverse;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<cum_sum>::save(ob);
        ob << axis;
        ob << exclusive;
        ob << reverse;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<cum_sum>::load(ib);
        ib >> axis;
        ib >> exclusive;
        ib >> reverse;
    }
};
}  // namespace cldnn
