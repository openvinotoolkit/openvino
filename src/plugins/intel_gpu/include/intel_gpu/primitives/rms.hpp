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
    /// @param axis Axis along which normalization is performed (default -1)
    rms(const primitive_id& id, const input_info& input, const input_info& gamma, const float epsilon, const int64_t axis = -1)
        : primitive_base(id, {input, gamma}),
          epsilon(epsilon),
          elementwise_affine(true),
          axis(axis) {}

    /// @brief Constructs rms primitive without gamma
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param epsilon Epsilon for not dividing by zero while normalizing
    /// @param axis Axis along which normalization is performed (default -1)
    rms(const primitive_id& id, const input_info& input, const float epsilon, const int64_t axis = -1)
        : primitive_base(id, {input}),
          epsilon(epsilon),
          elementwise_affine(false),
          axis(axis) {}

    /// @brief Epsilon for not dividing by zero while normalizing
    float epsilon;
    /// @brief A boolean value that when set to True, RMS has learnable affine parameters (gamma)
    bool elementwise_affine;
    /// @brief Normalization axis
    int64_t axis{-1};

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, epsilon);
        seed = hash_combine(seed, elementwise_affine);
        seed = hash_combine(seed, axis);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }

        auto rhs_casted = downcast<const rms>(rhs);

        return epsilon == rhs_casted.epsilon && elementwise_affine == rhs_casted.elementwise_affine && axis == rhs_casted.axis;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<rms>::save(ob);
        ob << epsilon;
        ob << elementwise_affine;
        ob << axis;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<rms>::load(ib);
        ib >> epsilon;
        ib >> elementwise_affine;
        ib >> axis;
    }
};
}  // namespace cldnn
