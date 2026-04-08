// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "primitive.hpp"

namespace cldnn {

using PagedCausalConv1D = ov::op::internal::PagedCausalConv1D;

struct paged_causal_conv1d : public primitive_base<paged_causal_conv1d> {
    CLDNN_DECLARE_PRIMITIVE(paged_causal_conv1d)

    enum PagedCausalConv1DInputIdx {
        INPUT_EMBEDS = 0,
        CONV_STATE_TABLE = 1,
        CONV_WEIGHT = 2,
        CONV_BIAS = 3,
        SUBSEQUENCE_BEGINS = 4,
        BLOCK_INDICES = 5,
        BLOCK_INDICES_BEGINS = 6,
        PAST_LENS = 7,
        CACHE_INTERVAL = 8,
    };

    paged_causal_conv1d() : primitive_base("", {}) {}

    paged_causal_conv1d(const primitive_id& id, const std::vector<input_info>& inputs) : primitive_base(id, inputs) {
        OPENVINO_ASSERT((inputs.size() == 9),
                        "[GPU] Unexpected inputs number for paged_causal_conv1d primitive: ",
                        inputs.size());
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, hidden_size);
        seed = hash_combine(seed, kernel_size);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const paged_causal_conv1d>(rhs);
        return hidden_size == rhs_casted.hidden_size && kernel_size == rhs_casted.kernel_size;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<paged_causal_conv1d>::save(ob);
        ob << hidden_size;
        ob << kernel_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<paged_causal_conv1d>::load(ib);
        ib >> hidden_size;
        ib >> kernel_size;
    }

    size_t hidden_size = 0;
    size_t kernel_size = 0;
};

}  // namespace cldnn
