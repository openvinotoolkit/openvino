// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {

struct stateless_kv : public primitive_base<stateless_kv> {
    CLDNN_DECLARE_PRIMITIVE(stateless_kv)

    stateless_kv() : primitive_base("", {}) {}

    stateless_kv(const primitive_id& id, const std::vector<input_info>& inputs, const int64_t concat_axis, const bool is_present_len)
        : primitive_base(id, inputs),
          concat_axis(concat_axis),
          is_present_len(is_present_len) {}

    int64_t concat_axis = 0;
    bool is_present_len = true;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, concat_axis);
        seed = hash_combine(seed, is_present_len);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const stateless_kv>(rhs);

        return concat_axis == rhs_casted.concat_axis && is_present_len == rhs_casted.is_present_len;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<stateless_kv>::save(ob);
        ob << concat_axis;
        ob << is_present_len;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<stateless_kv>::load(ib);
        ib >> concat_axis;
        ib >> is_present_len;
    }
};
}  // namespace cldnn
