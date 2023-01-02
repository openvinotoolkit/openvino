// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

#include "openvino/core/shape.hpp"

namespace cldnn {

/// @brief
/// @details
struct gather : public primitive_base<gather> {
    CLDNN_DECLARE_PRIMITIVE(gather)

    /// @brief Constructs gather primitive.
    /// @param id This primitive id.
    /// @param dict Input dictionary primitive id.
    /// @param idx Input indexes primitive id.
    /// @param axis Gathering axis.
    /// @param output_shape Output shape.
    /// @param batch_dim Batch_dim
    /// @param support_neg_ind Support negative indexes
    gather(const primitive_id& id,
           const input_info& dict,
           const input_info& idx,
           const int64_t axis,
           const ov::Shape& output_shape,
           const int64_t batch_dim = 0,
           const bool support_neg_ind = false,
           const padding& output_padding = padding())
        : primitive_base(id, {dict, idx}, {output_padding})
        , axis(axis)
        , output_shape(output_shape)
        , batch_dim(batch_dim)
        , support_neg_ind(support_neg_ind) {}

    /// @brief Gathering axis
    int64_t axis;
    /// @brief Gather output shape
    ov::Shape output_shape;
    /// @brief Gathering batch_dim
    int64_t batch_dim;
    /// @brief Support negative indexes
    bool support_neg_ind;

    size_t hash() const override {
        if (!seed) {
            seed = hash_combine(seed, axis);
            seed = hash_combine(seed, batch_dim);
            seed = hash_combine(seed, support_neg_ind);
        }
        return seed;
    }
};
}  // namespace cldnn
