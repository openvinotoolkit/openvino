// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Swish Gated Linear Unit Activation primitive
/// @details Performs gated linear unit activation that combines swish activation function
struct swiglu : public primitive_base<swiglu> {
    CLDNN_DECLARE_PRIMITIVE(swiglu);

    swiglu() : primitive_base("", {}) {}

    /// @brief Constructs swiglu primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param axis The index of an axis in data along which to perform the split
    /// @param split_lengths A list containing the sizes of each output tensor along the split axis
    /// @param output_size Output data size of the primitive
    swiglu(const primitive_id& id,
           const input_info& input,
           const int64_t& axis,
           const int64_t& split_lengths,
           const tensor output_size,
           const padding& output_padding = padding())
           : primitive_base(id, {input}, {output_padding}),
             axis(axis),
             split_lengths(split_lengths),
             output_size(output_size) {}

    int64_t axis = 0;
    int64_t split_lengths = 0;
    tensor output_size;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, axis);
        seed = hash_combine(seed, split_lengths);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const swiglu>(rhs);
        return axis == rhs_casted.axis && split_lengths == rhs_casted.split_lengths;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<swiglu>::save(ob);
        ob << axis;
        ob << split_lengths;
        ob << output_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<swiglu>::load(ib);
        ib >> axis;
        ib >> split_lengths;
        ib >> output_size;
    }
};
}  // namespace cldnn
