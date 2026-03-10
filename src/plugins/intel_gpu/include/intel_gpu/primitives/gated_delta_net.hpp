// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include <vector>

namespace cldnn {

using GatedDeltaNet = ov::op::GatedDeltaNet;

/// @brief gated_delta_net primitive
/// @details Performs gated_delta_net
struct gated_delta_net : public primitive_base<gated_delta_net> {
    CLDNN_DECLARE_PRIMITIVE(gated_delta_net)

    gated_delta_net() : primitive_base("", {}) {}

    /// @brief Constructs gated_delta_net primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    gated_delta_net(const primitive_id& id,
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
        primitive_base<gated_delta_net>::save(ob);
        // ob << k_head_size;
        // ob << v_head_size;
        // ob << k_heads_num;
        // ob << v_heads_num;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gated_delta_net>::load(ib);
        // ib >> k_head_size;
        // ib >> v_head_size;
        // ib >> k_heads_num;
        // ib >> v_heads_num;
    }

    // size_t k_head_size = 0;
    // size_t v_head_size = 0;
    // size_t k_heads_num = 0;
    // size_t v_heads_num = 0;
};

}  // namespace cldnn
