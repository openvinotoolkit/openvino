// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief Permutes data in the memory, with respect to provided order.
/// @details Permute order is set as vector with positions meaning corresponding to tensor.
/// Vector values represent dimensions to be permuted in bfyx format. For example: <br>
/// input_dimensions = tensor{ 5, 3, 6, 3 } <br>
/// permute_order = { 2, 3, 1, 0 } <br>
/// output_dimensions = { 6, 3, 3, 5 } <br>
/// <br>
/// When permute_order is { 0, 1, 2, 3 } then input_dimensions = output_dimensions
struct permute : public primitive_base<permute> {
    CLDNN_DECLARE_PRIMITIVE(permute)

    permute() : primitive_base("", {}) {}

    /// @brief Constructs permute primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param permute_order Array of permuted output order in bfyx format.
    permute(const primitive_id& id,
            const input_info& input,
            const std::vector<uint16_t>& permute_order = {})
        : primitive_base(id, {input}, 1, {optional_data_type()}), permute_order(permute_order) { }

    /// @brief Array of permuted output order in bfyx format.
    std::vector<uint16_t> permute_order;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, permute_order.begin(), permute_order.end());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const permute>(rhs);

        return permute_order == rhs_casted.permute_order;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<permute>::save(ob);
        ob << permute_order;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<permute>::load(ib);
        ib >> permute_order;
    }
};
}  // namespace cldnn
