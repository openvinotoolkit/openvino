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

    selective_ssm(const primitive_id& id, const std::vector<input_info>& inputs) : primitive_base(id, inputs) {
        OPENVINO_ASSERT(inputs.size() == 6, "Unexpected inputs number for selective_ssm primitive: ", inputs.size());
        num_outputs = 2;
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<selective_ssm>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<selective_ssm>::load(ib);
    }
};

}  // namespace cldnn
