// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "intel_gpu/op/moe_router_fused.hpp"
#include "primitive.hpp"

namespace cldnn {
using MoERouterFused = ov::intel_gpu::op::MoERouterFused;

/// @brief moe_router_fused primitive
/// @details Performs expert routing (softmax/sigmoid + top-k + normalization).
///
/// Inputs:
///   0: router_logits  [num_tokens, num_experts]
///   1: routing_bias   (optional, SIGMOID_BIAS only) [1, num_experts]
///   2: routing_eps    (optional, SIGMOID_BIAS only) scalar
///
/// Outputs:
///   0: topk_weights   [num_tokens, top_k] (f16/f32)
///   1: topk_indices   [num_tokens, top_k] (u32)
struct moe_router_fused : public primitive_base<moe_router_fused> {
    CLDNN_DECLARE_PRIMITIVE(moe_router_fused)

    moe_router_fused() : primitive_base("", {}) {}

    moe_router_fused(const primitive_id& id,
                     const std::vector<input_info>& inputs,
                     const MoERouterFused::Config& config)
        : primitive_base(id, inputs, 2, {optional_data_type(), optional_data_type()}),
          _config(config) {}

    MoERouterFused::Config _config;

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const moe_router_fused>(rhs);

        return _config.num_expert == rhs_casted._config.num_expert &&
               _config.top_k == rhs_casted._config.top_k &&
               _config.routing_type == rhs_casted._config.routing_type;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_router_fused>::save(ob);
        ob << make_data(&_config, sizeof(_config));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_router_fused>::load(ib);
        ib >> make_data(&_config, sizeof(_config));
    }
};

}  // namespace cldnn
