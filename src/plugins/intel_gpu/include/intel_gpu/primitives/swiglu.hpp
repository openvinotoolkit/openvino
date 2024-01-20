// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief SwiGLU activation primitive
struct swiglu : public primitive_base<swiglu> {
    CLDNN_DECLARE_PRIMITIVE(swiglu);

    swiglu() : primitive_base("", {}) {}

    /// @brief Constructs swiglu primitive
    swiglu(const primitive_id& id,
           const input_info& input,
           const int32_t axis,
           const int32_t split_length,
           const tensor output_size,
           const padding& output_padding = padding())
           : primitive_base(id, {input}, {output_padding}),
             axis(axis),
             split_length(split_length),
             output_size(output_size) {}

    int32_t axis;
    int32_t split_length;
    tensor output_size;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, axis);
        seed = hash_combine(seed, split_length);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const swiglu>(rhs);
        return axis == rhs_casted.axis && split_length == rhs_casted.split_length;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<swiglu>::save(ob);
        ob << axis;
        ob << split_length;
        ob << output_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<swiglu>::load(ib);
        ib >> axis;
        ib >> split_length;
        ib >> output_size;
    }
};
}  // namespace cldnn
