// Copyright (C) 2023 Intel Corporation
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
          epsilon(epsilon) {}

    /// @brief Epsilon for not dividing by zero while normalizing
    float epsilon;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, epsilon);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const rms>(rhs);

        return epsilon == rhs_casted.epsilon;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<rms>::save(ob);
        ob << epsilon;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<rms>::load(ib);
        ib >> epsilon;
    }
};
}  // namespace cldnn
