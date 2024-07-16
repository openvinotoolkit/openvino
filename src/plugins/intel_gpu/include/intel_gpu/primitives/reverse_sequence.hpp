// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct reverse_sequence : public primitive_base<reverse_sequence> {
    CLDNN_DECLARE_PRIMITIVE(reverse_sequence)

    reverse_sequence() : primitive_base("", {}) {}

    /// @brief Constructs reverse_sequence primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param seq_lengths Sequence lengths primitive id.
    /// @param seq_axis The axis which is partially reversed.
    /// @param batch_axis The axis along which reversal is performed.
    reverse_sequence(const primitive_id& id,
                     const input_info& input,
                     const input_info& seq_lengths,
                     const int32_t seq_axis,
                     const int32_t batch_axis = 0)
        : primitive_base(id, {input, seq_lengths}), seq_axis(seq_axis), batch_axis(batch_axis) {
        const int32_t number_of_dims = 4;

        int32_t batch_a = batch_axis;
        int32_t seq_a = seq_axis;

        if (batch_a < 0)
            batch_a += number_of_dims;

        if (seq_a < 0)
            seq_a += number_of_dims;

        if (batch_a == seq_a)
            throw std::runtime_error("Batch axis and sequence axis should not be equal\n");

        if (batch_a < 0 || batch_a >= number_of_dims)
            throw std::runtime_error("Incorrect batch axis value! Actual axis is" + std::to_string(batch_a));

        if (seq_a < 0 || seq_a >= number_of_dims)
            throw std::runtime_error("Incorrect sequence axis value! Actual axis is" + std::to_string(seq_a));
    }

    /// @brief The axis which is partially reversed.
    int32_t seq_axis = 0;
    /// @brief The axis along which reversal is performed.
    int32_t batch_axis = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, seq_axis);
        seed = hash_combine(seed, batch_axis);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const reverse_sequence>(rhs);

        return seq_axis == rhs_casted.seq_axis &&
               batch_axis == rhs_casted.batch_axis;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<reverse_sequence>::save(ob);
        ob << seq_axis;
        ob << batch_axis;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<reverse_sequence>::load(ib);
        ib >> seq_axis;
        ib >> batch_axis;
    }
};
}  // namespace cldnn
