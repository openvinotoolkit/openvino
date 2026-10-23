// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "primitive.hpp"

namespace cldnn {

/// @brief Multi-scale deformable attention, the GPU counterpart of ov::op::internal::MSDA.
/// @details Inputs: value, value_spatial_shapes, level_start_index, sampling_locations and attention_weights.
struct msda : public primitive_base<msda> {
    CLDNN_DECLARE_PRIMITIVE(msda)

    msda() : primitive_base("", {}) {}

    /// @brief Constructs msda primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    msda(const primitive_id& id, const std::vector<input_info>& inputs) : primitive_base(id, inputs) {}

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<msda>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<msda>::load(ib);
    }
};

}  // namespace cldnn
