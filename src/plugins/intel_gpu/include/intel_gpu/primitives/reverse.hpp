// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {

enum class reverse_mode : uint32_t { index, mask };

struct reverse : public primitive_base<reverse> {
    CLDNN_DECLARE_PRIMITIVE(reverse)

    reverse() : primitive_base("", {}) {}

    /// @brief Constructs reverse primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param axes Axes to reverse primitive id.
    /// @param mode Axes interpretation mode (indices/mask).
    reverse(const primitive_id& id,
            const input_info& input,
            const input_info& axes,
            const reverse_mode mode)
        : primitive_base{id, {input, axes}},
          mode{mode} {}

    reverse_mode mode{reverse_mode::index};

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, mode);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const reverse>(rhs);

        return mode == rhs_casted.mode;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<reverse>::save(ob);
        ob << make_data(&mode, sizeof(reverse_mode));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<reverse>::load(ib);
        ib >> make_data(&mode, sizeof(reverse_mode));
    }
};
}  // namespace cldnn
