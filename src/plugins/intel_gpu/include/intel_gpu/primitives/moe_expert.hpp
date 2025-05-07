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
            const scale_zp_mems& scale_zp)
        : primitive_base(id, inputs, 1, {optional_data_type()}),
          _config(config),
          _mlp_params(param),
          _scale_zp(scale_zp) {
    }

    size_t get_hidden_size() const {
        auto shape = _mlp_params[0].param[0].weight->get_layout().get_shape();

        if (shape.size() == 3) {
            return shape[1] * shape[2];
        }
        OPENVINO_ASSERT(shape.size() == 2);
        return shape[1];
    }

    size_t get_intermediate_size() const {
        auto shape = _mlp_params[0].param[0].weight->get_layout().get_shape();

        return shape[0];
    }

    size_t get_group_size() const {
        auto shape = _mlp_params[0].param[0].weight->get_layout().get_shape();
        if (shape.size() == 3) {
            return shape[2];
        }
        return shape[1];
    }

    MOEExpert::Config _config;
    std::vector<mlp_params> _mlp_params;
    scale_zp_mems _scale_zp;

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
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_expert>::load(ib);
        ib >> _config.expert_num;
        ib >> _config.hidden_size;
        ib >> _config.topk;
    }

protected:
    std::vector<input_info> get_dependencies() const override { return {}; }
};

}  // namespace cldnn
