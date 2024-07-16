// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct scatter_nd_update : public primitive_base<scatter_nd_update> {
    CLDNN_DECLARE_PRIMITIVE(scatter_nd_update)

    scatter_nd_update() : primitive_base("", {}) {}

    /// @brief Constructs scatter_nd_update primitive.
    /// @param id This primitive id.
    /// @param dict Input data primitive id.
    /// @param idx Input indexes primitive id.
    /// @param idupd Input updates primitive id.
    /// @param indices_rank Rank of indices.
    scatter_nd_update(const primitive_id& id,
                      const input_info& data,
                      const input_info& idx,
                      const input_info& idupd,
                      const size_t indices_rank)
        : primitive_base(id, {data, idx, idupd}), indices_rank(indices_rank) {}

    /// @brief ScatterNDUpdate indices_rank
    size_t indices_rank = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, indices_rank);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const scatter_nd_update>(rhs);

        return indices_rank == rhs_casted.indices_rank;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<scatter_nd_update>::save(ob);
        ob << indices_rank;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<scatter_nd_update>::load(ib);
        ib >> indices_rank;
    }
};
}  // namespace cldnn
