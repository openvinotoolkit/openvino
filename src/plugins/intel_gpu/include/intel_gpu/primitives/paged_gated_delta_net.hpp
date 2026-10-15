// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "primitive.hpp"

namespace cldnn {

using PagedGatedDeltaNet = ov::op::internal::PagedGatedDeltaNet;

/// @brief paged_gated_delta_net primitive
/// @details Performs paged gated_delta_net.
struct paged_gated_delta_net : public primitive_base<paged_gated_delta_net> {
    CLDNN_DECLARE_PRIMITIVE(paged_gated_delta_net)

    enum PagedGatedDeltaNetInputIdx {
        QUERY = 0,
        KEY = 1,
        VALUE = 2,
        RECURRENT_STATE_TABLE = 3,
        GATE = 4,
        BETA = 5,
        SUBSEQUENCE_BEGINS = 6,
        BLOCK_INDICES = 7,
        BLOCK_INDICES_BEGINS = 8,
        PAST_LENS = 9,
        CACHE_INTERVAL = 10,
    };

    paged_gated_delta_net() : primitive_base("", {}) {}

    paged_gated_delta_net(const primitive_id& id,
                          const std::vector<input_info>& inputs,
                          bool use_qk_l2norm = false,
                          float q_l2_norm_eps = 1e-6f,
                          float k_l2_norm_eps = 1e-6f)
        : primitive_base(id, inputs),
          use_qk_l2norm(use_qk_l2norm),
          q_l2_norm_eps(q_l2_norm_eps),
          k_l2_norm_eps(k_l2_norm_eps) {
        OPENVINO_ASSERT((inputs.size() == 11), "[GPU] Unexpected inputs number for paged_gated_delta_net primitive: ", inputs.size());
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, k_head_size);
        seed = hash_combine(seed, v_head_size);
        seed = hash_combine(seed, k_heads_num);
        seed = hash_combine(seed, v_heads_num);
        seed = hash_combine(seed, use_qk_l2norm);
        seed = hash_combine(seed, q_l2_norm_eps);
        seed = hash_combine(seed, k_l2_norm_eps);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const paged_gated_delta_net>(rhs);
        return k_head_size == rhs_casted.k_head_size && v_head_size == rhs_casted.v_head_size && k_heads_num == rhs_casted.k_heads_num &&
               v_heads_num == rhs_casted.v_heads_num && use_qk_l2norm == rhs_casted.use_qk_l2norm && q_l2_norm_eps == rhs_casted.q_l2_norm_eps &&
               k_l2_norm_eps == rhs_casted.k_l2_norm_eps;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<paged_gated_delta_net>::save(ob);
        ob << k_head_size;
        ob << v_head_size;
        ob << k_heads_num;
        ob << v_heads_num;
        ob << use_qk_l2norm;
        ob << q_l2_norm_eps;
        ob << k_l2_norm_eps;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<paged_gated_delta_net>::load(ib);
        ib >> k_head_size;
        ib >> v_head_size;
        ib >> k_heads_num;
        ib >> v_heads_num;
        ib >> use_qk_l2norm;
        ib >> q_l2_norm_eps;
        ib >> k_l2_norm_eps;
    }

    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t k_heads_num = 0;
    size_t v_heads_num = 0;
    bool use_qk_l2norm = false;
    float q_l2_norm_eps = 1e-6f;
    float k_l2_norm_eps = 1e-6f;
};

}  // namespace cldnn
