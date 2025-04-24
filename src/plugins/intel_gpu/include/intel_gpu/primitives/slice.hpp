// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct slice : public primitive_base<slice> {
    CLDNN_DECLARE_PRIMITIVE(slice)

    slice() : primitive_base("", {}) {}

    /// @brief Constructs slice primitive.
    /// @param id This primitive id.
    /// @param inputs List of primitive ids.
    slice(const primitive_id& id,
          const std::vector<input_info>& inputs)
        : primitive_base{id, inputs}
    {}

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<slice>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<slice>::load(ib);
    }
};
}  // namespace cldnn
