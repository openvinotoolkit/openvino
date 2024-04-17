// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct scaled_dot_product_attention : public primitive_base<scaled_dot_product_attention> {
    CLDNN_DECLARE_PRIMITIVE(scaled_dot_product_attention)

    scaled_dot_product_attention() : primitive_base("", {}) {}

    /// @brief Constructs scaled_dot_product_attention primitive.
    /// @param id This primitive id.
    /// @param inputq, inputk, inputv inputs of ScaledDotProductAttention.
    scaled_dot_product_attention(const primitive_id& id,
        const input_info& inputq, const input_info& inputk, const input_info& inputv,
        const input_info& scale,
        const padding& output_padding = padding())
        : primitive_base(id, {inputq, inputk, inputv, scale}, {output_padding})
        {}

    /// @brief Constructs scaled_dot_product_attention primitive.
    /// @param id This primitive id.
    /// @param inputq, inputk, inputv inputs of ScaledDotProductAttention.
    scaled_dot_product_attention(const primitive_id& id,
        const input_info& inputq, const input_info& inputk, const input_info& inputv,
        const input_info& scale, const input_info& attention_mask,
        const padding& output_padding = padding())
        : primitive_base(id, {inputq, inputk, inputv, scale, attention_mask}, {output_padding})
        {}

    size_t hash() const override {
        size_t seed = primitive::hash();
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        return true;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<scaled_dot_product_attention>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<scaled_dot_product_attention>::load(ib);
    }
};
}  // namespace cldnn
