// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Computes a histogram of all input elements using equal-width bins (Histc-17).
struct histc : public primitive_base<histc> {
    CLDNN_DECLARE_PRIMITIVE(histc)

    histc() : primitive_base("", {}) {}

    /// @brief Constructs histc primitive.
    /// @param id This primitive id.
    /// @param input Input data.
    /// @param bins Number of equal-width bins.
    /// @param min Lower end of the range. If equal to max, the range is taken from the data.
    /// @param max Upper end of the range. If equal to min, the range is taken from the data.
    histc(const primitive_id& id, const input_info& input, int64_t bins, double min, double max)
        : primitive_base(id, {input}),
          bins(bins),
          min(min),
          max(max) {}

    int64_t bins = 100;
    double min = 0.0;
    double max = 0.0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, bins);
        seed = hash_combine(seed, min);
        seed = hash_combine(seed, max);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const histc>(rhs);
        return bins == rhs_casted.bins && min == rhs_casted.min && max == rhs_casted.max;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<histc>::save(ob);
        ob << bins;
        ob << min;
        ob << max;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<histc>::load(ib);
        ib >> bins;
        ib >> min;
        ib >> max;
    }
};
}  // namespace cldnn
