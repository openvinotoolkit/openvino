// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>

#include "primitive.hpp"

namespace cldnn {

/// @brief Elementwise y = clamp(x * scale + bias, lo, hi). Experimental.
struct scaled_shifted_clamp_experimental : public primitive_base<scaled_shifted_clamp_experimental> {
    CLDNN_DECLARE_PRIMITIVE(scaled_shifted_clamp_experimental);

    scaled_shifted_clamp_experimental() : primitive_base("", {}) {}

    scaled_shifted_clamp_experimental(const primitive_id& id,
                                      const input_info& input,
                                      const float scale,
                                      const float bias,
                                      const float lo,
                                      const float hi)
        : primitive_base(id, {input}),
          scale(scale),
          bias(bias),
          lo(lo),
          hi(hi) {}

    float scale{1.0f};
    float bias{0.0f};
    float lo{std::numeric_limits<float>::lowest()};
    float hi{std::numeric_limits<float>::max()};

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, scale);
        seed = hash_combine(seed, bias);
        seed = hash_combine(seed, lo);
        seed = hash_combine(seed, hi);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }
        auto rhs_casted = downcast<const scaled_shifted_clamp_experimental>(rhs);
        return scale == rhs_casted.scale && bias == rhs_casted.bias && lo == rhs_casted.lo && hi == rhs_casted.hi;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<scaled_shifted_clamp_experimental>::save(ob);
        ob << scale;
        ob << bias;
        ob << lo;
        ob << hi;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<scaled_shifted_clamp_experimental>::load(ib);
        ib >> scale;
        ib >> bias;
        ib >> lo;
        ib >> hi;
    }
};

}  // namespace cldnn
