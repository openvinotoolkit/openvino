// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"
#include "ov_ops/moe_expert.hpp"
#include <vector>

namespace cldnn {
using MOEExpert = ov::op::internal::MOEExpert;

/// @brief moe_expert primitive
/// @details Performs moe expert
struct moe_expert : public primitive_base<moe_expert> {
    CLDNN_DECLARE_PRIMITIVE(moe_expert)

    moe_expert() : primitive_base("", {}) {}

    struct mlp_params {
        struct param {
            cldnn::memory::ptr weight;
            cldnn::memory::ptr bias;
            cldnn::memory::ptr scale;
            cldnn::memory::ptr zp;
            // ba layout copy
            cldnn::memory::ptr scale_ba;
            cldnn::memory::ptr zp_ba;
            bool operator==(const param& rhs) const {
                return weight == rhs.weight && bias == rhs.bias && scale == rhs.scale && zp == rhs.zp;
            }
        } param[3];

        bool operator==(const mlp_params& rhs) const {
            return param[0] == rhs.param[0] && param[1] == rhs.param[1] && param[2] == rhs.param[2];
        }
    };

    #define EACH_EXPERT_WEIGHTS_OFFSET_SIZE 64
    struct mlp_weights_mem {
        memory::ptr weights_base;
        // weights/scale/zp offsets, each expert has 9*4 = 36 bytes
        // gate_weight_offset, up_weight_offset, down_weight_offset
        // gate_scale_offset, up_scale_offset, down_scale_offset
        // gate_zp_offset, up_zp_offset, down_zp_offset
        memory::ptr weights_offset;
    };

    struct scale_zp_mems {
        // [64bytes]->gate_addrs,up_addrs, gate_scales_addrs, up_scales_addrs,gate_zp_addrs ,up_zp_addrs, padding1, padding2
        memory::ptr gate_up_addrs;
        memory::ptr down_addrs;
        memory::ptr down_scales_addrs;
        memory::ptr down_zp_addrs;    
    };

    /// @brief Constructs moe_expert primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    moe_expert(const primitive_id& id,
            const std::vector<input_info>& inputs,
            const MOEExpert::Config& config, const std::vector<mlp_params>& param,
            const mlp_weights_mem& wei_mem,
            const scale_zp_mems& scale_zp,
            const uint8_t cm_mask)
        : primitive_base(id, inputs, 1, {optional_data_type()}),
          _config(config),
          _mlp_params(param),
          _mlp_weights_mem(wei_mem),
          _scale_zp(scale_zp),
          _cm_mask(cm_mask) {
    }

    MOEExpert::Config _config;
    std::vector<mlp_params> _mlp_params;
    mlp_weights_mem _mlp_weights_mem;
    scale_zp_mems _scale_zp;
    uint8_t _cm_mask;

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const moe_expert>(rhs);

        return std::memcmp(&_config, &rhs_casted._config, sizeof(_config)) == 0 &&
               _mlp_params == rhs_casted._mlp_params;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_expert>::save(ob);
        ob << _config.expert_num;
        ob << _config.hidden_size;
        ob << _config.topk;
        ob << _config.fused_router_logic;
        ob << _config.intermediate_size;
        ob << _config.group_size;
        // ob << _config.weight_type;
        // ob << _config.scale_type;
        // ob << _config.zp_type;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_expert>::load(ib);
        ib >> _config.expert_num;
        ib >> _config.hidden_size;
        ib >> _config.topk;
        ib >> _config.fused_router_logic;
        ib >> _config.intermediate_size;
        ib >> _config.group_size;
        // ib >> _config.weight_type;
        // ib >> _config.scale_type;
        // ib >> _config.zp_type;
    }

protected:
    std::vector<input_info> get_dependencies() const override { return {}; }
};

}  // namespace cldnn
