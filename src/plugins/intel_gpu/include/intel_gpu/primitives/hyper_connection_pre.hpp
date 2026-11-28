// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "openvino/op/hyper_connection_pre.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief mHC pre side (ov::op::internal::HyperConnectionPre): streams [..., n, e], fn [(2+n)*n, n*e], scale [3],
/// base [(2+n)*n] -> y [..., e], post [..., n], comb [..., n, n] ([src][dst]).
struct hyper_connection_pre : public primitive_base<hyper_connection_pre> {
    CLDNN_DECLARE_PRIMITIVE(hyper_connection_pre)

    hyper_connection_pre() : primitive_base("", {}) {}

    hyper_connection_pre(const primitive_id& id,
                         const std::vector<input_info>& inputs,
                         int64_t n_streams,
                         int64_t sinkhorn_iterations,
                         float eps,
                         float rms_eps)
        : primitive_base(id, inputs, 3),
          n_streams(n_streams),
          sinkhorn_iterations(sinkhorn_iterations),
          eps(eps),
          rms_eps(rms_eps) {}

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, n_streams);
        seed = hash_combine(seed, sinkhorn_iterations);
        seed = hash_combine(seed, eps);
        seed = hash_combine(seed, rms_eps);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }
        auto rhs_casted = downcast<const hyper_connection_pre>(rhs);
        return n_streams == rhs_casted.n_streams && sinkhorn_iterations == rhs_casted.sinkhorn_iterations &&
               eps == rhs_casted.eps && rms_eps == rhs_casted.rms_eps;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<hyper_connection_pre>::save(ob);
        ob << n_streams;
        ob << sinkhorn_iterations;
        ob << eps;
        ob << rms_eps;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<hyper_connection_pre>::load(ib);
        ib >> n_streams;
        ib >> sinkhorn_iterations;
        ib >> eps;
        ib >> rms_eps;
    }

    int64_t n_streams = 4;
    int64_t sinkhorn_iterations = 20;
    float eps = 1e-6F;
    float rms_eps = 1e-6F;
};

}  // namespace cldnn
