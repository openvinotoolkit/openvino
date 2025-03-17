// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief feed_forward primitive
struct feed_forward : public primitive_base<feed_forward> {
    CLDNN_DECLARE_PRIMITIVE(feed_forward);

    feed_forward() : primitive_base("", {}) {}

    /// @brief Constructs feed_forward primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param input2 Input2 primitive id
    /// @param input3 Input3 primitive id
    /// @param input4 Input4 primitive id
    /// @param input5 Input5 primitive id
    /// @param output_size Output data size of the primitive
    feed_forward(const primitive_id& id,
           const input_info& input,
           const input_info& input2,
           const input_info& input3,
           const input_info& input4,
           const input_info& input5,
           const tensor output_size)
           : primitive_base(id, {input, input2, input3, input4, input5}),
             output_size(output_size) {}

    tensor output_size;

    size_t hash() const override {
        size_t seed = primitive::hash();
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        return true;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<feed_forward>::save(ob);

    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<feed_forward>::load(ib);
        ib >> output_size;
    }
};
}  // namespace cldnn
