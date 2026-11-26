// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "openvino/op/hyper_connection_post.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief mHC post step (ov::op::internal::HyperConnectionPost): y [..., E], residual [..., n, E], post [..., n],
/// comb [..., n, n] ([src][dst]) -> [..., n, E].
struct hyper_connection_post : public primitive_base<hyper_connection_post> {
    CLDNN_DECLARE_PRIMITIVE(hyper_connection_post)

    hyper_connection_post() : primitive_base("", {}) {}

    hyper_connection_post(const primitive_id& id, const std::vector<input_info>& inputs) : primitive_base(id, inputs) {}

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<hyper_connection_post>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<hyper_connection_post>::load(ib);
    }
};

}  // namespace cldnn
