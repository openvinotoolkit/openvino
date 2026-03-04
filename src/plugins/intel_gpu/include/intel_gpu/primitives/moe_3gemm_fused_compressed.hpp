// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>
#include <string>

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "primitive.hpp"

namespace cldnn {
using MOE3GemmFusedCompressed = ov::intel_gpu::op::MOE3GemmFusedCompressed;
extern size_t lru_expert_num;

struct moe_weights {
    cldnn::memory::ptr gate_w = nullptr;
    cldnn::memory::ptr gate_s = nullptr;
    cldnn::memory::ptr gate_z = nullptr;
    cldnn::memory::ptr up_w = nullptr;
    cldnn::memory::ptr up_s = nullptr;
    cldnn::memory::ptr up_z = nullptr;
    cldnn::memory::ptr down_w = nullptr;
    cldnn::memory::ptr down_s = nullptr;
    cldnn::memory::ptr down_z = nullptr;
};

/// @brief moe compressed primitive
/// @details Performs moe compressed
struct moe_3gemm_fused_compressed : public primitive_base<moe_3gemm_fused_compressed> {
    CLDNN_DECLARE_PRIMITIVE(moe_3gemm_fused_compressed)

    static constexpr size_t serialized_weight_offset_count = 9;

    enum class input_index : size_t {
        hidden_states = 0,
        routing_weights,
        weight_0,
        scale_0,
        zp_0,
        weight_1,
        scale_1,
        zp_1,
        weight_2,
        scale_2,
        zp_2,
        count
    };
    static constexpr size_t input_count = static_cast<size_t>(input_index::count);

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
        moe_3gemm_fused_compressed(const primitive_id& id,
                                                             const std::vector<input_info>& inputs,
                                                             const MOE3GemmFusedCompressed::Config& config,
                                                             const std::vector<size_t>& weight_bin_offsets = {},
                                                             const std::string& weights_path = "")
        : primitive_base(id, inputs, 1, {optional_data_type()}),
                    _config(config),
                    _weight_bin_offsets(weight_bin_offsets),
                    _weights_path(weights_path) {}

    MOE3GemmFusedCompressed::Config _config;
        std::vector<size_t> _weight_bin_offsets;
        std::string _weights_path;

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const moe_3gemm_fused_compressed>(rhs);

         return std::memcmp(&_config, &rhs_casted._config, sizeof(_config)) == 0 &&
             _weight_bin_offsets == rhs_casted._weight_bin_offsets &&
             _weights_path == rhs_casted._weights_path;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_3gemm_fused_compressed>::save(ob);
        ob << make_data(&_config, sizeof(_config));
        ob << _weight_bin_offsets;
        ob << _weights_path;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_3gemm_fused_compressed>::load(ib);
        ib >> make_data(&_config, sizeof(_config));
        ib >> _weight_bin_offsets;
        ib >> _weights_path;
    }
};

}  // namespace cldnn
