// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct shuffle_channels : public primitive_base<shuffle_channels> {
    CLDNN_DECLARE_PRIMITIVE(shuffle_channels)

    shuffle_channels() : primitive_base("", {}) {}

    /// @brief Constructs shuffle_channels primitive.
    /// @param id This primitive id.
    /// @param input Input dictionary primitive id.
    /// @param group The number of groups to split the channel dimension.
    /// @param axis The index of the channel dimension.
    shuffle_channels(const primitive_id& id,
                     const input_info& input,
                     const int32_t group,
                     const int32_t axis = 1)
        : primitive_base(id, {input}), group(group), axis(axis) {}

    /// @brief The number of groups to split the channel dimension. This number must evenly divide the channel dimension size.
    int32_t group;
    /// @brief The index of the channel dimension (default is 1).
    int32_t axis = 1;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, group);
        seed = hash_combine(seed, axis);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const shuffle_channels>(rhs);

        return group == rhs_casted.group &&
               axis == rhs_casted.axis;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<shuffle_channels>::save(ob);
        ob << group;
        ob << axis;
   }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<shuffle_channels>::load(ib);
        ib >> group;
        ib >> axis;
    }
};
}  // namespace cldnn
