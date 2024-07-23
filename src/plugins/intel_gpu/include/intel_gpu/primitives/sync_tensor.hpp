// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct sync_tensor : public primitive_base<sync_tensor> {
    CLDNN_DECLARE_PRIMITIVE(sync_tensor)

    sync_tensor() : primitive_base("", {}) {}

    /// @brief Constructs sync_tensor primitive.
    /// @param id This primitive id.
    /// @param inputs of sync_tensor.
    sync_tensor(const primitive_id& id,
                const input_info& input)
        : primitive_base(id, {input})
        {}

    size_t hash() const override {
        size_t seed = primitive::hash();
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        return true;
    }

    /* We don't have any argument to serialize at this moment. */
    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<sync_tensor>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<sync_tensor>::load(ib);
    }
    layout output_layout;
};
}  // namespace cldnn