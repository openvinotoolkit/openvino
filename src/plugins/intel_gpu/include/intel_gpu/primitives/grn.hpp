// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Global Response Normalization primitive.
struct grn : public primitive_base<grn> {
    CLDNN_DECLARE_PRIMITIVE(grn)

    /// @brief Constructs grn primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param bias Bias value for whole output tensor.
    grn(const primitive_id& id,
        const input_info& input,
        const float bias,
        const data_types data_type,
        const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}, {optional_data_type{ data_type }}),
        bias(bias)
    {}

    /// @brief Bias value for whole output tensor.
    float bias;

    size_t hash() const override {
        if (!seed) {
            seed = hash_combine(seed, bias);
        }
        return seed;
    }
};
}  // namespace cldnn
