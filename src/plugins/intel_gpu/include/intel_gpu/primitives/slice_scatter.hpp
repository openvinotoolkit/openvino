// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief SliceScatter operation.
/// @details Creates a copy of `data` tensor and updates it with `updates` at positions
///          specified by start/stop/step/axes slicing parameters.
struct slice_scatter : public primitive_base<slice_scatter> {
    CLDNN_DECLARE_PRIMITIVE(slice_scatter)

    slice_scatter() : primitive_base("", {}) {}

    /// @brief Constructs slice_scatter primitive.
    /// @param id This primitive id.
    /// @param inputs List of primitive ids: data, updates, start, stop, step, [axes].
    slice_scatter(const primitive_id& id,
                  const std::vector<input_info>& inputs)
        : primitive_base{id, inputs}
    {}

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<slice_scatter>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<slice_scatter>::load(ib);
    }
};
}  // namespace cldnn
