// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"
#include "openvino/op/fused_conv.hpp"
#include <vector>

namespace cldnn {

using FusedConv = ov::op::FusedConv;

/// @brief fused_conv primitive
/// @details Fuses Gather(beam_idx) + Concat + GroupConv + SiLU + Slice for depthwise causal conv
struct fused_conv : public primitive_base<fused_conv> {
    CLDNN_DECLARE_PRIMITIVE(fused_conv)

    fused_conv() : primitive_base("", {}) {}

    /// @brief Constructs fused_conv primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    fused_conv(const primitive_id& id,
            const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<fused_conv>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<fused_conv>::load(ib);
    }
};

}  // namespace cldnn
