// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "primitive.hpp"

namespace cldnn {

using PagedSelectiveSSM = ov::op::internal::PagedSelectiveSSM;

struct paged_selective_ssm : public primitive_base<paged_selective_ssm> {
    CLDNN_DECLARE_PRIMITIVE(paged_selective_ssm)

    enum PagedSelectiveSSMInputIdx {
        A = 0,
        DT = 1,
        B = 2,
        X = 3,
        C = 4,
        RECURRENT_STATE_TABLE = 5,
        SUBSEQUENCE_BEGINS = 6,
        BLOCK_INDICES = 7,
        BLOCK_INDICES_BEGINS = 8,
        NUM_PROCESSED_TOKENS = 9,
        CACHE_INTERVAL = 10,
    };

    paged_selective_ssm() : primitive_base("", {}) {}

    paged_selective_ssm(const primitive_id& id, const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {
        OPENVINO_ASSERT(inputs.size() == 11, "Unexpected inputs number for paged_selective_ssm primitive: ", inputs.size());
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_heads);
        seed = hash_combine(seed, num_groups);
        seed = hash_combine(seed, head_dim);
        seed = hash_combine(seed, state_size);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        const auto& rhs_casted = downcast<const paged_selective_ssm>(rhs);
        return num_heads == rhs_casted.num_heads && num_groups == rhs_casted.num_groups &&
               head_dim == rhs_casted.head_dim && state_size == rhs_casted.state_size;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<paged_selective_ssm>::save(ob);
        ob << num_heads;
        ob << num_groups;
        ob << head_dim;
        ob << state_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<paged_selective_ssm>::load(ib);
        ib >> num_heads;
        ib >> num_groups;
        ib >> head_dim;
        ib >> state_size;
    }

    size_t num_heads = 0;
    size_t num_groups = 0;
    size_t head_dim = 0;
    size_t state_size = 0;
};

}  // namespace cldnn
