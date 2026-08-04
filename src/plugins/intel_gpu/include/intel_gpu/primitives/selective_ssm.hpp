// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "primitive.hpp"

namespace cldnn {

using SelectiveSSM = ov::op::internal::SelectiveSSM;

struct selective_ssm : public primitive_base<selective_ssm> {
    CLDNN_DECLARE_PRIMITIVE(selective_ssm)

    selective_ssm() : primitive_base("", {}) {}

    selective_ssm(const primitive_id& id, const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {}

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

        const auto& rhs_casted = downcast<const selective_ssm>(rhs);
        return num_heads == rhs_casted.num_heads && num_groups == rhs_casted.num_groups &&
               head_dim == rhs_casted.head_dim && state_size == rhs_casted.state_size;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<selective_ssm>::save(ob);
        ob << num_heads;
        ob << num_groups;
        ob << head_dim;
        ob << state_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<selective_ssm>::load(ib);
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
