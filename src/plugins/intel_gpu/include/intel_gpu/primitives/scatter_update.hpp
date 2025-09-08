// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct scatter_update : public primitive_base<scatter_update> {
    CLDNN_DECLARE_PRIMITIVE(scatter_update)

    scatter_update() : primitive_base("", {}) {}

    enum scatter_update_axis {
        along_b,
        along_f,
        along_x,
        along_y,
        along_z,
        along_w
    };

    /// @brief Constructs scatter_update primitive.
    /// @param id This primitive id.
    /// @param dict Input dictionary primitive id.
    /// @param idx Input indexes primitive id.
    /// @param idupd Input updates primitive id.
    /// @param axis Gathering axis.
    scatter_update(const primitive_id& id,
                   const input_info& dict,
                   const input_info& idx,
                   const input_info& idupd,
                   const int64_t axis)
        : primitive_base(id, {dict, idx, idupd}), axis(axis) {}

    /// @brief ScatterUpdate axis
    int64_t axis = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, axis);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const scatter_update>(rhs);

        return axis == rhs_casted.axis;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<scatter_update>::save(ob);
        ob << axis;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<scatter_update>::load(ib);
        ib >> axis;
    }
};
}  // namespace cldnn
