// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace tp_gpu {
namespace op {

/// @brief Tensor-parallel Gather: collects every rank's slice of a tensor into rank 0.
class TPGather : public ov::op::Op {
public:
    OPENVINO_OP("TPGather", "tp_gpu", Op);

    TPGather() = default;

    /// \param data          This rank's slice.
    /// \param collective_id Which exchange this is.
    ///                      The ranks meet by it, so it has to be the same one on every rank.
    /// \param rank          This rank's index.
    ///                      Only rank 0's output grows to hold every slice, the others keep their own.
    /// \param world_size    How many slices there are.
    /// \param axis          Axis the slices are concatenated along; negative counts from the end.
    TPGather(const Output<Node>& data,
             uint32_t collective_id,
             uint32_t rank,
             uint32_t world_size,
             int64_t axis = -1);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    uint32_t get_collective_id() const { return m_collective_id; }
    uint32_t get_rank() const { return m_rank; }
    uint32_t get_world_size() const { return m_world_size; }
    int64_t get_axis() const { return m_axis; }

private:
    uint32_t m_collective_id = 0;
    uint32_t m_rank = 0;
    uint32_t m_world_size = 1;
    int64_t m_axis = -1;
};

}  // namespace op
}  // namespace tp_gpu
}  // namespace ov
