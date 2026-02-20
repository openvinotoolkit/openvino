// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Root Mean Square Normalization primitive
/// @details Performs re-scaling invariance and regularizes the summed input according to RMS statistics
struct rms : public primitive_base<rms> {
    CLDNN_DECLARE_PRIMITIVE(rms);

    rms() : primitive_base("", {}) {}

    /// @brief Constructs rms primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param gamma Gamma values for weight
    /// @param epsilon Epsilon for not dividing by zero while normalizing
    rms(const primitive_id& id,
        const input_info& input,
        const input_info& gamma,
        const float epsilon)
        : primitive_base(id, {input, gamma}),
          epsilon(epsilon),
          elementwise_affine(true) {}

    /// @brief Constructs rms primitive without gamma
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param epsilon Epsilon for not dividing by zero while normalizing
    rms(const primitive_id& id,
        const input_info& input,
        const float epsilon)
        : primitive_base(id, {input}),
          epsilon(epsilon),
          elementwise_affine(false) {}

    /// @brief Epsilon for not dividing by zero while normalizing
    float epsilon;
    /// @brief A boolean value that when set to True, RMS has learnable affine parameters (gamma)
    bool elementwise_affine;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, epsilon);
        seed = hash_combine(seed, elementwise_affine);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const rms>(rhs);

        return epsilon == rhs_casted.epsilon &&
               elementwise_affine == rhs_casted.elementwise_affine;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<rms>::save(ob);
        ob << epsilon;
        ob << elementwise_affine;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<rms>::load(ib);
        ib >> epsilon;
        ib >> elementwise_affine;
    }
};
}  // namespace cldnn
