// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "primitive.hpp"

namespace cldnn {

struct unique_count : primitive_base<unique_count> {
    CLDNN_DECLARE_PRIMITIVE(unique_count)

    unique_count() : primitive_base("", {}) {}

    /// @brief Constructs unique_count primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param flattened If true, operator works on a flattened version of the input tensor.
    /// @param axis Is used to “divide” the input tensor into slices.
    unique_count(const primitive_id& id, const input_info& input, bool flattened, int64_t axis)
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
        auto rhs_casted = downcast<const unique_count>(rhs);
        return flattened == rhs_casted.flattened && axis == rhs_casted.axis;
    }
};

struct unique_gather : primitive_base<unique_gather> {
    CLDNN_DECLARE_PRIMITIVE(unique_gather)

    unique_gather() : primitive_base("", {}) {}

    /// @brief Constructs unique_gather primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param flattened If true, operator works on a flattened version of the input tensor.
    /// @param axis Is used to “divide” the input tensor into slices.
    /// @param sorted Controls the order of the returned unique values (sorts ascending when true).
    unique_gather(const primitive_id& id,
                  const std::vector<input_info>& inputs,
                  bool flattened,
                  int64_t axis,
                  bool sorted,
                  data_types elem_type,
                  data_types index_type,
                  data_types count_type)
        : primitive_base(id, inputs, 4, {elem_type, index_type, index_type, count_type}),
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
        auto rhs_casted = downcast<const unique_gather>(rhs);
        return flattened == rhs_casted.flattened && axis == rhs_casted.axis && sorted == rhs_casted.sorted;
    }
};

}  // namespace cldnn
