// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "primitive.hpp"

namespace cldnn {

/// @brief FusedMLP primitive (POC).
/// @details Computes SwiGLU MLP block using weights provided as inputs.
/// Inputs:
///   0: X
///   1: W_gate
///   2: W_up
///   3: W_down
struct fused_mlp : public primitive_base<fused_mlp> {
    CLDNN_DECLARE_PRIMITIVE(fused_mlp)

    fused_mlp() : primitive_base("", {}) {}

    fused_mlp(const primitive_id& id, const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {}

    size_t hash() const override {
        return primitive::hash();
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }
};

}  // namespace cldnn

