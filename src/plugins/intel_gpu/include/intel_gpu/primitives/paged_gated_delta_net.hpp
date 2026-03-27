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
                          const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {
        OPENVINO_ASSERT((inputs.size() == 11),
                        "[GPU] Unexpected inputs number for paged_gated_delta_net primitive: ",
                        inputs.size());
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, k_head_size);
        seed = hash_combine(seed, v_head_size);
        seed = hash_combine(seed, k_heads_num);
        seed = hash_combine(seed, v_heads_num);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const paged_gated_delta_net>(rhs);
        return k_head_size == rhs_casted.k_head_size &&
               v_head_size == rhs_casted.v_head_size &&
               k_heads_num == rhs_casted.k_heads_num &&
               v_heads_num == rhs_casted.v_heads_num;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<paged_gated_delta_net>::save(ob);
        ob << k_head_size;
        ob << v_head_size;
        ob << k_heads_num;
        ob << v_heads_num;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<paged_gated_delta_net>::load(ib);
        ib >> k_head_size;
        ib >> v_head_size;
        ib >> k_heads_num;
        ib >> v_heads_num;
    }

    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t k_heads_num = 0;
    size_t v_heads_num = 0;
};

}  // namespace cldnn
