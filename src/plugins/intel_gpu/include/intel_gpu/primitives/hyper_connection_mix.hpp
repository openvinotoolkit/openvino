// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "openvino/op/hyper_connection_mix.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief mHC coefficient op (ov::op::internal::HyperConnectionMix): mixes [..., (2+n)*n], scale [3], base [(2+n)*n]
/// -> pre [..., n], post [..., n], comb [..., n, n] ([src][dst]).
struct hyper_connection_mix : public primitive_base<hyper_connection_mix> {
    CLDNN_DECLARE_PRIMITIVE(hyper_connection_mix)

    hyper_connection_mix() : primitive_base("", {}) {}

    hyper_connection_mix(const primitive_id& id, const std::vector<input_info>& inputs, int64_t n_streams, int64_t sinkhorn_iterations, float eps)
        : primitive_base(id, inputs, 3),
          n_streams(n_streams),
          sinkhorn_iterations(sinkhorn_iterations),
          eps(eps) {}

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, n_streams);
        seed = hash_combine(seed, sinkhorn_iterations);
        seed = hash_combine(seed, eps);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }
        auto rhs_casted = downcast<const hyper_connection_mix>(rhs);
        return n_streams == rhs_casted.n_streams && sinkhorn_iterations == rhs_casted.sinkhorn_iterations && eps == rhs_casted.eps;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<hyper_connection_mix>::save(ob);
        ob << n_streams;
        ob << sinkhorn_iterations;
        ob << eps;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<hyper_connection_mix>::load(ib);
        ib >> n_streams;
        ib >> sinkhorn_iterations;
        ib >> eps;
    }

    int64_t n_streams = 4;
    int64_t sinkhorn_iterations = 20;
    float eps = 1e-6F;
};

}  // namespace cldnn
