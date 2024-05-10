// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

struct scaled_dot_product_attention : public primitive_base<scaled_dot_product_attention> {
    CLDNN_DECLARE_PRIMITIVE(scaled_dot_product_attention)

    scaled_dot_product_attention() : primitive_base("", {}) {}

    /// @brief Constructs scaled_dot_product_attention primitive.
    /// @param id This primitive id.
    /// @param inputs Input data primitives id (query, keys, values, [attention_mask], [scale]).
    /// @param is_causal If true, assumes causal attention masking. In this case attention_mask input is ignored.
    scaled_dot_product_attention(const primitive_id& id,
                                 const std::vector<cldnn::input_info> inputs,
                                 bool is_causal,
                                 const padding& output_padding = padding())
        : primitive_base(id, inputs, {output_padding})
        , is_causal(is_causal)
        , has_attn_mask_input(inputs.size() > 3)
        , has_scale_input(inputs.size() > 4) {}

    bool is_causal = false;
    bool has_attn_mask_input = false;
    bool has_scale_input = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, is_causal);
        seed = hash_combine(seed, has_attn_mask_input);
        seed = hash_combine(seed, has_scale_input);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const scaled_dot_product_attention>(rhs);

        return is_causal == rhs_casted.is_causal &&
               has_attn_mask_input == rhs_casted.has_attn_mask_input &&
               has_scale_input == rhs_casted.has_scale_input;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<scaled_dot_product_attention>::save(ob);
        ob << is_causal;
        ob << has_attn_mask_input;
        ob << has_scale_input;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<scaled_dot_product_attention>::load(ib);
        ib >> is_causal;
        ib >> has_attn_mask_input;
        ib >> has_scale_input;
    }
};
}  // namespace cldnn
