// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace tp_gpu {
namespace op {

/// @brief Tensor-parallel Gather: collects every rank's slice of a tensor into
/// rank 0.
///
/// Deliberately a gather and not an all-gather.  The only consumer of the
/// gathered tensor is the model's Result, and the infer request reads results
/// from rank 0 alone, so delivering the full tensor to every rank would move
/// world_size-1 times the data for copies nobody reads.
///
/// Shapes follow from that: on rank 0 the gathered axis grows by world_size,
/// on every other rank the output keeps the rank's own slice and is left
/// untouched at execution time.
class TPGather : public ov::op::Op {
public:
    OPENVINO_OP("TPGather", "tp_gpu", Op);

    TPGather() = default;

    /// \param data          this rank's slice.
    /// \param group_id      collective group the ranks belong to.
    /// \param collective_id slot identifying this gather within the group.
    /// \param rank          this rank's index.
    /// \param world_size    number of ranks in the group.
    /// \param axis          axis the slices are concatenated along; negative
    ///                      counts from the end.
    TPGather(const Output<Node>& data,
             uint32_t group_id,
             uint32_t collective_id,
             uint32_t rank,
             uint32_t world_size,
             int64_t axis = -1);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    uint32_t get_group_id() const { return m_group_id; }
    uint32_t get_collective_id() const { return m_collective_id; }
    uint32_t get_rank() const { return m_rank; }
    uint32_t get_world_size() const { return m_world_size; }
    int64_t get_axis() const { return m_axis; }

private:
    uint32_t m_group_id = 0;
    uint32_t m_collective_id = 0;
    uint32_t m_rank = 0;
    uint32_t m_world_size = 1;
    int64_t m_axis = -1;
};

}  // namespace op
}  // namespace tp_gpu
}  // namespace ov
