// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "primitive.hpp"

namespace cldnn {

struct histc : primitive_base<histc> {
    CLDNN_DECLARE_PRIMITIVE(histc)

    histc() : primitive_base("", {}) {}

    histc(const primitive_id& id,
          const std::vector<input_info>& inputs,
          data_types output_type,
          int64_t bins = 100,
          double min_val = 0.0,
          double max_val = 0.0)
        : primitive_base(id, inputs, 1, {optional_data_type(output_type)}),
          bins(bins),
          min_val(min_val),
          max_val(max_val) {}

    int64_t bins = 100;
    double min_val = 0.0;
    double max_val = 0.0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, bins);
        seed = hash_combine(seed, min_val);
        seed = hash_combine(seed, max_val);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const histc>(rhs);
        return bins == rhs_casted.bins && min_val == rhs_casted.min_val && max_val == rhs_casted.max_val;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<histc>::save(ob);
        ob << bins;
        ob << min_val;
        ob << max_val;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<histc>::load(ib);
        ib >> bins;
        ib >> min_val;
        ib >> max_val;
    }
};

}  // namespace cldnn
