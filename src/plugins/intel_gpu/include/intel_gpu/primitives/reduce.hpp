// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief Select mode for the @ref reduce layer
enum class reduce_mode : uint16_t {
    /// @brief Reduce max
    max,
    /// @brief Reduce min
    min,
    /// @brief Reduce mean
    mean,
    /// @brief Reduce prod
    prod,
    /// @brief Reduce sum
    sum,
    /// @brief Reduce and
    logical_and,
    /// @brief Reduce or
    logical_or,
    /// @brief Reduce  sum_square
    sum_square,
    /// @brief Reduce l1
    l1,
    /// @brief Reduce l2
    l2,
    /// @brief Reduce log_sum
    log_sum,
    /// @brief Reduce sum_exp
    log_sum_exp
};

/// @brief Applies the specific reduction function along provided axes (second input) of the input tensor (first input).
/// @details
struct reduce : public primitive_base<reduce> {
    CLDNN_DECLARE_PRIMITIVE(reduce)

    reduce() : primitive_base("", {}) {}

    /// @brief Constructs reduce primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param keep_dims The axes which reduced
    reduce(const primitive_id& id,
           const input_info& input,
           const reduce_mode mode,
           const std::vector<int64_t> axes,
           const bool keep_dims)
        : primitive_base(id, {input}), mode(mode), axes(axes), keep_dims(keep_dims) {}

    /// @brief Reduce operation type
    reduce_mode mode;
    /// @brief List of axes to reduce
    std::vector<int64_t> axes;
    /// @brief Keep the reduced dimension or not, 1 mean keep reduced dimension
    bool keep_dims = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, mode);
        seed = hash_range(seed, axes.begin(), axes.end());
        seed = hash_combine(seed, keep_dims);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const reduce>(rhs);

        return mode == rhs_casted.mode &&
               axes == rhs_casted.axes &&
               keep_dims == rhs_casted.keep_dims;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<reduce>::save(ob);
        ob << make_data(&mode, sizeof(reduce_mode));
        ob << axes;
        ob << keep_dims;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<reduce>::load(ib);
        ib >> make_data(&mode, sizeof(reduce_mode));
        ib >> axes;
        ib >> keep_dims;
    }
};
}  // namespace cldnn
