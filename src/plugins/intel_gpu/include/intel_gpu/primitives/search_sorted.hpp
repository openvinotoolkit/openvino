// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct search_sorted : public primitive_base<search_sorted> {
    CLDNN_DECLARE_PRIMITIVE(search_sorted)

    search_sorted() : primitive_base("", {}) {}

    /// @brief Constructs search_sorted primitive.
    /// @param id This primitive id.
    /// @param sorted Sorted input.
    /// @param values Values input.
    /// @param right_mode Enable/Disable right mode(check specification for details)..
    search_sorted(const primitive_id& id, const input_info& sorted, const input_info& values, bool right_mode)
        : primitive_base(id, {sorted, values}),
          right_mode(right_mode) {}

    /// @brief Enable/Disable right mode(check specification for details).
    bool right_mode = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, right_mode);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const search_sorted>(rhs);

        return right_mode == rhs_casted.right_mode;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<search_sorted>::save(ob);
        ob << right_mode;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<search_sorted>::load(ib);
        ib >> right_mode;
    }
};
}  // namespace cldnn
