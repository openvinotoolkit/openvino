// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Global Response Normalization primitive.
struct grn : public primitive_base<grn> {
    CLDNN_DECLARE_PRIMITIVE(grn)

    grn() : primitive_base("", {}) {}

    /// @brief Constructs grn primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param bias Bias value for whole output tensor.
    grn(const primitive_id& id,
        const input_info& input,
        const float bias,
        const data_types data_type)
        : primitive_base(id, {input}, 1, {optional_data_type{ data_type }}),
        bias(bias)
    {}

    /// @brief Bias value for whole output tensor.
    float bias = 0.0f;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, bias);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const grn>(rhs);

        return bias == rhs_casted.bias;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<grn>::save(ob);
        ob << bias;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<grn>::load(ib);
        ib >> bias;
    }
};
}  // namespace cldnn
