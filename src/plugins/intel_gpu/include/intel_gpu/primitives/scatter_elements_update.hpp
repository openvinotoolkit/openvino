// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct scatter_elements_update : public primitive_base<scatter_elements_update> {
    CLDNN_DECLARE_PRIMITIVE(scatter_elements_update)

    /// @brief Constructs scatter_elements_update primitive.
    /// @param id This primitive id.
    /// @param dict Input data primitive id.
    /// @param idx Input indexes primitive id.
    /// @param idupd Input updates primitive id.
    /// @param axis Gathering axis.
    scatter_elements_update(const primitive_id& id,
                            const input_info& data,
                            const input_info& idx,
                            const input_info& idupd,
                            const int64_t axis,
                            const padding& output_padding = padding())
        : primitive_base(id, {data, idx, idupd}, {output_padding}), axis(axis) {}

    /// @brief ScatterElementsUpdate axis
    int64_t axis;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, axis);
        return seed;
    }
};
}  // namespace cldnn
