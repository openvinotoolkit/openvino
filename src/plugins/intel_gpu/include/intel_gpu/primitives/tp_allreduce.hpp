// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

#include <memory>
#include <string>

namespace ov { namespace tp_gpu { class TPCoordination; } }

namespace cldnn {

/// @brief Tensor-parallel AllReduce: sums a tensor across TP ranks via host-side coordination.
/// Implemented as a CPU primitive in the GPU plugin — waits for GPU, reads to host,
/// coordinates with peer ranks via a shared TPCoordination object, writes result back.
struct tp_allreduce : public primitive_base<tp_allreduce> {
    CLDNN_DECLARE_PRIMITIVE(tp_allreduce)

    tp_allreduce() : primitive_base("", {}) {}

    tp_allreduce(const primitive_id& id,
                 const input_info& input,
                 uint32_t collective_id,
                 uint32_t rank,
                 std::shared_ptr<ov::tp_gpu::TPCoordination> coordination)
        : primitive_base(id, {input}),
          collective_id(collective_id),
          rank(rank),
          coordination(std::move(coordination)) {}

    uint32_t collective_id = 0;
    uint32_t rank = 0;
    std::shared_ptr<ov::tp_gpu::TPCoordination> coordination;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, collective_id);
        seed = hash_combine(seed, rank);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const tp_allreduce>(rhs);
        return collective_id == rhs_casted.collective_id &&
               rank == rhs_casted.rank;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<tp_allreduce>::save(ob);
        ob << collective_id;
        ob << rank;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<tp_allreduce>::load(ib);
        ib >> collective_id;
        ib >> rank;
    }
};

}  // namespace cldnn
