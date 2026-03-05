// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "primitive.hpp"

namespace cldnn {
using MOE3GemmFusedCompressed = ov::intel_gpu::op::MOE3GemmFusedCompressed;

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
    //                   1: routing_weights - [num_seq, num_experts] routing weights for all experts
    //                   2: w0_weight - expert weights for first projection,
    //                      shape [num_experts, inter_size, group_num, group_size]
    //                   3: w0_scale - expert scale for first projection for compressed experts,
    //                      shape [num_experts, inter_size, group_num, 1]
    //                   4: w0_zp - expert zp for first projection for compressed experts,
    //                      shape [num_experts, inter_size, group_num, 1]
    //                   5: w1_weight - expert weights for second projection,
    //                      shape [num_experts, inter_size, group_num, group_size]
    //                   6: w1_scale - expert scale for second projection for compressed experts,
    //                      shape [num_experts, inter_size, group_num, 1]
    //                   7: w1_zp - expert zp for second projection for compressed experts,
    //                      shape [num_experts, inter_size, group_num, 1]
    //                   8: w2_weight - expert weights for final projection,
    //                      shape [num_experts, hidden_size, group_num, group_size]
    //                   9: w2_scale - expert scale for final projection for compressed experts,
    //                      shape [num_experts, hidden_size, group_num, 1]
    //                   10: w2_zp - expert zp for final projection for compressed experts,
    //
    moe_3gemm_fused_compressed(const primitive_id& id, const std::vector<input_info>& inputs, const MOE3GemmFusedCompressed::Config& config)
        : primitive_base(id, inputs, 1, {optional_data_type()}),
          _config(config) {}

    MOE3GemmFusedCompressed::Config _config;

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
