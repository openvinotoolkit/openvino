// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "primitive.hpp"

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
    /// @param config             GatedDeltaNet config struct
    gated_delta_net(const primitive_id& id, const std::vector<input_info>& inputs, const GatedDeltaNet::Config& config)
        : primitive_base(id, inputs),
          config(config) {}

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, k_head_size);
        seed = hash_combine(seed, v_head_size);
        seed = hash_combine(seed, k_heads_num);
        seed = hash_combine(seed, v_heads_num);
        seed = hash_combine(seed, config.fuse_qk_l2norm);
        seed = hash_combine(seed, config.q_l2_norm_eps);
        seed = hash_combine(seed, config.k_l2_norm_eps);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const gated_delta_net>(rhs);
        return k_head_size == rhs_casted.k_head_size && v_head_size == rhs_casted.v_head_size && k_heads_num == rhs_casted.k_heads_num &&
               v_heads_num == rhs_casted.v_heads_num && config.fuse_qk_l2norm == rhs_casted.config.fuse_qk_l2norm &&
               config.q_l2_norm_eps == rhs_casted.config.q_l2_norm_eps && config.k_l2_norm_eps == rhs_casted.config.k_l2_norm_eps;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gated_delta_net>::save(ob);
        ob << k_head_size;
        ob << v_head_size;
        ob << k_heads_num;
        ob << v_heads_num;
        ob << config.fuse_qk_l2norm;
        ob << config.q_l2_norm_eps;
        ob << config.k_l2_norm_eps;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gated_delta_net>::load(ib);
        ib >> k_head_size;
        ib >> v_head_size;
        ib >> k_heads_num;
        ib >> v_heads_num;
        ib >> config.fuse_qk_l2norm;
        ib >> config.q_l2_norm_eps;
        ib >> config.k_l2_norm_eps;
    }

    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t k_heads_num = 0;
    size_t v_heads_num = 0;
    GatedDeltaNet::Config config;
};

}  // namespace cldnn
