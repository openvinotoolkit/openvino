// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "primitive.hpp"

namespace cldnn {

/// @brief Unique-10 primitive.
struct unique : primitive_base<unique> {
    CLDNN_DECLARE_PRIMITIVE(unique)

    /// @brief Constructs unique primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param flattened If true, operator works on a flattened version of the input tensor.
    /// @param axis Is used to “divide” the input tensor into slices.
    unique(const primitive_id& id, const input_info& input, bool flattened, int64_t axis)
        : primitive_base(id, {input}),
          flattened(flattened),
          axis(axis) {}

    bool flattened;
    int64_t axis;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, flattened);
        seed = hash_combine(seed, axis);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }
        auto rhs_casted = downcast<const unique>(rhs);
        return flattened == rhs_casted.flattened && axis == rhs_casted.axis;
    }
};

/// @brief Reshape unique outputs to match total unique count shape
struct unique_reshape : primitive_base<unique_reshape> {
    CLDNN_DECLARE_PRIMITIVE(unique_reshape)

    unique_reshape(const primitive_id& id,
                   const std::vector<input_info>& inputs,
                   bool flattened,
                   int64_t axis,
                   bool sorted,
                   data_types elem_type,
                   data_types index_type,
                   data_types count_type)
        : primitive_base(id, inputs, decltype(output_paddings)(4), {elem_type, index_type, index_type, count_type}, 4),
          flattened(flattened),
          axis(axis),
          sorted(sorted) {}

    bool flattened;
    int64_t axis;
    bool sorted;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, flattened);
        seed = hash_combine(seed, axis);
        seed = hash_combine(seed, sorted);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }
        auto rhs_casted = downcast<const unique_reshape>(rhs);
        return flattened == rhs_casted.flattened && axis == rhs_casted.axis && sorted == rhs_casted.sorted;
    }
};

}  // namespace cldnn
