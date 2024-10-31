// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <algorithm>
#include <vector>

#include "openvino/op/util/attr_types.hpp"
#include "primitive.hpp"

namespace cldnn {

struct search_sorted : public primitive_base<search_sorted> {
    CLDNN_DECLARE_PRIMITIVE(search_sorted)

    search_sorted() : primitive_base("", {}) {}

    size_t hash() const override {
        size_t seed = primitive::hash();
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<search_sorted>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<search_sorted>::load(ib);
    }
};
}  // namespace cldnn
