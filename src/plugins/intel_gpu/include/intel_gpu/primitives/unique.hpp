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
    /// @param sorted Controls the order of the returned unique values (sorts ascending when true).
    unique(const primitive_id& id,
           const input_info& input,
           bool flattened,
           int64_t axis,
           bool sorted,
           const std::vector<padding>& output_paddings,
           const std::vector<optional_data_type>& output_data_types,
           size_t num_outputs)
        : primitive_base(id, {input}, output_paddings, output_data_types, num_outputs),
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
        auto rhs_casted = downcast<const unique>(rhs);
        return flattened == rhs_casted.flattened && axis == rhs_casted.axis && sorted == rhs_casted.sorted;
    }
};

/// @brief Reshape unique outputs to match total unique count shape
struct unique_reshape : primitive_base<unique_reshape> {
    CLDNN_DECLARE_PRIMITIVE(unique_reshape)

    unique_reshape(const primitive_id& id,
                   const std::vector<input_info>& inputs,
                   bool flattened,
                   int64_t axis,
                   const std::vector<padding>& output_paddings,
                   const std::vector<optional_data_type>& output_data_types,
                   size_t num_outputs)
        : primitive_base(id, inputs, output_paddings, output_data_types, num_outputs),
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
        auto rhs_casted = downcast<const unique_reshape>(rhs);
        return flattened == rhs_casted.flattened && axis == rhs_casted.axis;
    }
};

}  // namespace cldnn
