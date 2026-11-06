// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {

struct stateless_kv : public primitive_base<stateless_kv> {
    CLDNN_DECLARE_PRIMITIVE(stateless_kv)

    stateless_kv() : primitive_base("", {}) {}

    stateless_kv(const primitive_id& id,
                 const std::vector<input_info>& inputs,
                 const int64_t concat_axis,
                 const bool is_seq_len_present_len,
                 const int64_t window_size = 0)
        : primitive_base(id, inputs),
          concat_axis(concat_axis),
          is_seq_len_present_len(is_seq_len_present_len),
          window_size(window_size) {
        OPENVINO_ASSERT(concat_axis >= 0, "[GPU] stateless_kv concat_axis must be normalized");
    }

    int64_t concat_axis = 0;
    bool is_seq_len_present_len = true;
    int64_t window_size = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, concat_axis);
        seed = hash_combine(seed, is_seq_len_present_len);
        seed = hash_combine(seed, window_size);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const stateless_kv>(rhs);

        return concat_axis == rhs_casted.concat_axis && is_seq_len_present_len == rhs_casted.is_seq_len_present_len && window_size == rhs_casted.window_size;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<stateless_kv>::save(ob);
        ob << concat_axis;
        ob << is_seq_len_present_len;
        ob << window_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<stateless_kv>::load(ib);
        ib >> concat_axis;
        ib >> is_seq_len_present_len;
        ib >> window_size;
    }
};
}  // namespace cldnn
