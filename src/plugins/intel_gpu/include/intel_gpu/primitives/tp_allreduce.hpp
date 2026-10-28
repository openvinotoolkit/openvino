// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

#include <string>

namespace cldnn {

/// @brief Tensor-parallel AllReduce: sums a tensor across the ranks of a
/// collective group.
///
/// Carries only the metadata that identifies the collective; the channel that
/// runs it is attached to the network when the model is bound to a TP context,
/// which keeps this primitive fully serializable.
struct tp_allreduce : public primitive_base<tp_allreduce> {
    CLDNN_DECLARE_PRIMITIVE(tp_allreduce)

    tp_allreduce() : primitive_base("", {}) {}

    tp_allreduce(const primitive_id& id,
                 const input_info& input,
                 uint32_t collective_id,
                 uint32_t rank,
                 uint32_t world_size)
        : primitive_base(id, {input}),
          collective_id(collective_id),
          rank(rank),
          world_size(world_size) {}

    uint32_t collective_id = 0;
    uint32_t rank = 0;
    uint32_t world_size = 1;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, collective_id);
        seed = hash_combine(seed, rank);
        seed = hash_combine(seed, world_size);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const tp_allreduce>(rhs);
        return collective_id == rhs_casted.collective_id &&
               rank == rhs_casted.rank &&
               world_size == rhs_casted.world_size;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<tp_allreduce>::save(ob);
        ob << collective_id;
        ob << rank;
        ob << world_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<tp_allreduce>::load(ib);
        ib >> collective_id;
        ib >> rank;
        ib >> world_size;
    }
};

}  // namespace cldnn
