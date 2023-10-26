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

    gather() : primitive_base("", {}) {}

    /// @brief Constructs gather primitive.
    /// @param id This primitive id.
    /// @param dict Input dictionary primitive id.
    /// @param idx Input indexes primitive id.
    /// @param axis Gathering axis.
    /// @param input_rank Input rank.
    /// @param output_shape Output shape.
    /// @param batch_dim Batch_dim
    /// @param support_neg_ind Support negative indexes
    gather(const primitive_id& id,
           const input_info& dict,
           const input_info& idx,
           const int64_t axis,
           const int64_t input_rank,
           const ov::Shape& output_shape,
           const int64_t batch_dim = 0,
           const bool support_neg_ind = false,
           const padding& output_padding = padding())
        : primitive_base(id, {dict, idx}, {output_padding})
        , axis(axis)
        , input_rank(input_rank)
        , output_shape(output_shape)
        , batch_dim(batch_dim)
        , support_neg_ind(support_neg_ind) {}

    /// @brief Gathering axis
    int64_t axis = 0;
    /// @brief Gather input rank
    int64_t input_rank;
    /// @brief Gather output shape
    ov::Shape output_shape;
    /// @brief Gathering batch_dim
    int64_t batch_dim = 0;
    /// @brief Support negative indexes
    bool support_neg_ind = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, axis);
        seed = hash_combine(seed, batch_dim);
        seed = hash_combine(seed, support_neg_ind);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const gather>(rhs);

        return axis == rhs_casted.axis &&
               batch_dim == rhs_casted.batch_dim &&
               support_neg_ind == rhs_casted.support_neg_ind;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gather>::save(ob);
        ob << axis;
        ob << input_rank;
        ob << output_shape;
        ob << batch_dim;
        ob << support_neg_ind;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gather>::load(ib);
        ib >> axis;
        ib >> input_rank;
        ib >> output_shape;
        ib >> batch_dim;
        ib >> support_neg_ind;
    }
};
}  // namespace cldnn
