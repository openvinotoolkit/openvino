// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

#include <string>

namespace cldnn {

/// @brief Tensor-parallel Gather: collects every rank's slice of a tensor into
/// rank 0's buffer.
///
/// Like tp_allreduce, this carries only the metadata identifying the
/// collective; the coordinator that runs it is resolved at execution time
/// through the network's registry, which keeps the primitive serializable.
///
/// `world_size` is needed here and not in tp_allreduce because it decides the
/// output shape on the root.
struct tp_gather : public primitive_base<tp_gather> {
    CLDNN_DECLARE_PRIMITIVE(tp_gather)

    tp_gather() : primitive_base("", {}) {}

    tp_gather(const primitive_id& id,
              const input_info& input,
              uint32_t group_id,
              uint32_t collective_id,
              uint32_t rank,
              uint32_t world_size,
              int64_t axis)
        : primitive_base(id, {input}),
          group_id(group_id),
          collective_id(collective_id),
          rank(rank),
          world_size(world_size),
          axis(axis) {}

    uint32_t group_id = 0;
    uint32_t collective_id = 0;
    uint32_t rank = 0;
    uint32_t world_size = 1;
    int64_t axis = -1;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, group_id);
        seed = hash_combine(seed, collective_id);
        seed = hash_combine(seed, rank);
        seed = hash_combine(seed, world_size);
        seed = hash_combine(seed, axis);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const tp_gather>(rhs);
        return group_id == rhs_casted.group_id && collective_id == rhs_casted.collective_id &&
               rank == rhs_casted.rank && world_size == rhs_casted.world_size &&
               axis == rhs_casted.axis;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<tp_gather>::save(ob);
        ob << group_id;
        ob << collective_id;
        ob << rank;
        ob << world_size;
        ob << axis;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<tp_gather>::load(ib);
        ib >> group_id;
        ib >> collective_id;
        ib >> rank;
        ib >> world_size;
        ib >> axis;
    }
};

}  // namespace cldnn
