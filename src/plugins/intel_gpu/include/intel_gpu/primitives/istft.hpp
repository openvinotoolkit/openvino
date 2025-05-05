// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Inverse short time fourier transform (ISTFT) operation.
/// @details Checks the specification for details.
struct ISTFT : public primitive_base<ISTFT> {
    CLDNN_DECLARE_PRIMITIVE(ISTFT)

    ISTFT() : primitive_base("", {}) {}

    /// @brief Constructs ISTFT primitive.
    /// @param id This primitive id.
    /// @param inputs List of input primitives(check specification for details).
    /// @param center Enable/Disable center(check specification for details).
    /// @param normalized Enable/Disable center(check specification for details).
    ISTFT(const primitive_id& id,
          const std::vector<input_info>& inputs,
          const bool center,
          const bool normalized)
        : primitive_base(id, inputs),
          center(center),
          normalized(normalized) {}

    /// @brief Enable/Disable center(check specification for details).
    bool center = false;
    /// @brief Enable/Disable normalized(check specification for details).
    bool normalized = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, center);
        seed = hash_combine(seed, normalized);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const ISTFT>(rhs);

        return center == rhs_casted.center && normalized == rhs_casted.normalized;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<ISTFT>::save(ob);
        ob << center;
        ob << normalized;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<ISTFT>::load(ib);
        ib >> center;
        ib >> normalized;
    }
};
}  // namespace cldnn
