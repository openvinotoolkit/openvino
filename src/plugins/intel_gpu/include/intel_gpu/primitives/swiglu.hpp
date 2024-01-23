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
           const std::vector<int64_t>& axis,
           const std::vector<int64_t>& split_lengths,
           const tensor output_size,
           const padding& output_padding = padding())
           : primitive_base(id, {input}, {output_padding}),
             axis(axis),
             split_lengths(split_lengths),
             output_size(output_size) {}

    std::vector<int64_t> axis;
    std::vector<int64_t> split_lengths;
    tensor output_size;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, axis.begin(), axis.end());
        seed = hash_range(seed, split_lengths.begin(), split_lengths.end());
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
