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

    paged_selective_ssm(const primitive_id& id, const std::vector<input_info>& inputs) : primitive_base(id, inputs) {
        OPENVINO_ASSERT(inputs.size() == 11, "Unexpected inputs number for paged_selective_ssm primitive: ", inputs.size());
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<paged_selective_ssm>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<paged_selective_ssm>::load(ib);
    }
};

}  // namespace cldnn
