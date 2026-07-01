// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "ov_ops/moe_compressed.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "primitive.hpp"

namespace cldnn {
using MOECompressed = ov::op::internal::MOECompressed;

/// @brief moe compressed primitive
/// @details Performs moe compressed
struct moe_3gemm_fused_compressed : public primitive_base<moe_3gemm_fused_compressed> {
    CLDNN_DECLARE_PRIMITIVE(moe_3gemm_fused_compressed)

    moe_3gemm_fused_compressed() : primitive_base("", {}) {}

    // @brief Constructs moe primitive / layer.
    //
    // @param id      An identifier of new primitive.
    // @param inputs  A list of Input primitive ids (inputs).
    //                   0: hidden_states - input tensor with hidden representations
    //                   1: topk_weights - [num_tokens, top_k] normalized routing weights from MoERouterFused
    //                   2: topk_indices - [num_tokens, top_k] expert indices from MoERouterFused
    //                   3: w0_weight - expert weights for first projection,
    //                      shape [num_experts, inter_size, group_num, group_size]
    //                   4: w0_scale - expert scale for first projection for compressed experts,
    //                      shape [num_experts, inter_size, group_num, 1]
    //                   5: w0_zp - expert zp for first projection for compressed experts,
    //                      shape [num_experts, inter_size, group_num, 1]
    //                   6: w1_weight - expert weights for second projection,
    //                      shape [num_experts, inter_size, group_num, group_size]
    //                   7: w1_scale - expert scale for second projection for compressed experts,
    //                      shape [num_experts, inter_size, group_num, 1]
    //                   8: w1_zp - expert zp for second projection for compressed experts,
    //                      shape [num_experts, inter_size, group_num, 1]
    //                   9: w2_weight - expert weights for final projection,
    //                      shape [num_experts, hidden_size, group_num, group_size]
    //                   10: w2_scale - expert scale for final projection for compressed experts,
    //                      shape [num_experts, hidden_size, group_num, 1]
    //                   11: w2_zp - expert zp for final projection for compressed experts,
    //                      shape [num_experts, hidden_size, group_num, 1]
    //
    //                   Options for shared experts (if config.num_shared_expert > 0, always starting at index 12):
    //                   12: shared_gate_weight - shared expert weights for first projection,
    //                      shape [1, inter_size, group_num, group_size]
    //                   13: shared_gate_scale - shared expert scale for first projection,
    //                      shape [1, inter_size, group_num, 1]
    //                   14: shared_gate_zp - shared expert zp for first projection,
    //                      shape [1, inter_size, group_num, 1]
    //                   15: shared_up_weight - shared expert weights for second projection,
    //                      shape [1, inter_size, group_num, group_size]
    //                   16: shared_up_scale - shared expert scale for second projection,
    //                      shape [1, inter_size, group_num, 1]
    //                   17: shared_up_zp - shared expert zp for second projection,
    //                      shape [1, inter_size, group_num, 1]
    //                   18: shared_down_weight - shared expert weights for final projection,
    //                      shape [1, hidden_size, group_num, group_size]
    //                   19: shared_down_scale - shared expert scale for final projection,
    //                      shape [1, hidden_size, group_num, 1]
    //                   20: shared_down_zp - shared expert zp for final projection,
    //                      shape [1, hidden_size, group_num, 1]
    //                   21: shared_gate_gate_weight - shared expert gate weight for gating,
    //                      shape [hidden_size]
    //
    moe_3gemm_fused_compressed(const primitive_id& id, const std::vector<input_info>& inputs, const MOECompressed::Config& config)
        : primitive_base(id, inputs, 1, {optional_data_type()}),
          _config(config) {}

    MOECompressed::Config _config;

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const moe_3gemm_fused_compressed>(rhs);

        return std::memcmp(&_config, &rhs_casted._config, sizeof(_config)) == 0;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_3gemm_fused_compressed>::save(ob);
        ob << make_data(&_config, sizeof(_config));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_3gemm_fused_compressed>::load(ib);
        ib >> make_data(&_config, sizeof(_config));
    }
};

}  // namespace cldnn
