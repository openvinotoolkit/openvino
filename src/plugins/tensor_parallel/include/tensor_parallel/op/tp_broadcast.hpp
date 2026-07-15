// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace tp {

/// \brief Broadcast collective: broadcasts a tensor from root_rank to all ranks.
/// Output shape == input shape.
class TPBroadcast : public ov::op::Op {
public:
    OPENVINO_OP("TPBroadcast", "tp_internal_opset", Op);

    TPBroadcast() = default;

    /// \param data           The tensor to broadcast
    /// \param group_id       Logical group identifier
    /// \param collective_id  Unique id for matching across ranks
    /// \param world_size     Number of participating ranks
    /// \param root_rank      Rank that owns the source data
    TPBroadcast(const Output<Node>& data,
                uint32_t group_id,
                uint32_t collective_id,
                uint32_t world_size,
                uint32_t root_rank);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    uint32_t get_group_id() const { return m_group_id; }
    uint32_t get_collective_id() const { return m_collective_id; }
    uint32_t get_world_size() const { return m_world_size; }
    uint32_t get_root_rank() const { return m_root_rank; }

private:
    uint32_t m_group_id = 0;
    uint32_t m_collective_id = 0;
    uint32_t m_world_size = 1;
    uint32_t m_root_rank = 0;
};

}  // namespace tp
}  // namespace op
}  // namespace ov
