// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/util/variable.hpp"
#include "primitive.hpp"

namespace cldnn {

using GatedDeltaNet = ov::op::internal::GatedDeltaNet;

/// @brief gated_delta_net primitive
/// @details Performs gated_delta_net
struct gated_delta_net : public primitive_base<gated_delta_net> {
    CLDNN_DECLARE_PRIMITIVE(gated_delta_net)

    gated_delta_net() : primitive_base("", {}) {}

    /// @brief Constructs gated_delta_net primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    /// @param fuse_qk_l2norm     A boolean to enable l2norm for variables q and k.
    /// @param q_l2_norm_eps      Epsilon value for q's l2 normalization computation.
    /// @param k_l2_norm_eps      Epsilon value for k's l2 normalization computation.
    gated_delta_net(const primitive_id& id,
                    const std::vector<input_info>& inputs,
                    bool fuse_qk_l2norm = false,
                    float q_l2_norm_eps = 1e-6f,
                    float k_l2_norm_eps = 1e-6f,
                    const ov::op::util::VariableInfo& variable_info = {})
        : primitive_base(id, inputs),
          fuse_qk_l2norm(fuse_qk_l2norm),
          q_l2_norm_eps(q_l2_norm_eps),
          k_l2_norm_eps(k_l2_norm_eps),
          variable_info(variable_info) {}

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, k_head_size);
        seed = hash_combine(seed, v_head_size);
        seed = hash_combine(seed, k_heads_num);
        seed = hash_combine(seed, v_heads_num);
        seed = hash_combine(seed, fuse_qk_l2norm);
        seed = hash_combine(seed, q_l2_norm_eps);
        seed = hash_combine(seed, k_l2_norm_eps);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const gated_delta_net>(rhs);
        return k_head_size == rhs_casted.k_head_size && v_head_size == rhs_casted.v_head_size && k_heads_num == rhs_casted.k_heads_num &&
               v_heads_num == rhs_casted.v_heads_num && fuse_qk_l2norm == rhs_casted.fuse_qk_l2norm && q_l2_norm_eps == rhs_casted.q_l2_norm_eps &&
               k_l2_norm_eps == rhs_casted.k_l2_norm_eps;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gated_delta_net>::save(ob);
        ob << k_head_size;
        ob << v_head_size;
        ob << k_heads_num;
        ob << v_heads_num;
        ob << fuse_qk_l2norm;
        ob << q_l2_norm_eps;
        ob << k_l2_norm_eps;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gated_delta_net>::load(ib);
        ib >> k_head_size;
        ib >> v_head_size;
        ib >> k_heads_num;
        ib >> v_heads_num;
        ib >> fuse_qk_l2norm;
        ib >> q_l2_norm_eps;
        ib >> k_l2_norm_eps;
    }

    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t k_heads_num = 0;
    size_t v_heads_num = 0;
    bool fuse_qk_l2norm = false;
    float q_l2_norm_eps = 1e-6f;
    float k_l2_norm_eps = 1e-6f;
    ov::op::util::VariableInfo variable_info;
};

}  // namespace cldnn
