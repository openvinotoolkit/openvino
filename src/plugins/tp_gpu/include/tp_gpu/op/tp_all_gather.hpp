// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace tp_gpu {
namespace op {

/// \brief All-gather collective: gathers tensor chunks from all ranks along an axis.
/// Output shape: input shape with dimension[axis] * world_size.
class TPAllGather : public ov::op::Op {
public:
    OPENVINO_OP("TPAllGather", "tp_gpu", Op);

    TPAllGather() = default;

    /// \param data           The local tensor chunk
    /// \param group_id       Logical group identifier
    /// \param collective_id  Unique id for matching across ranks
    /// \param world_size     Number of participating ranks
    /// \param axis           Axis along which to gather
    TPAllGather(const Output<Node>& data,
                uint32_t group_id,
                uint32_t collective_id,
                uint32_t world_size,
                int64_t axis);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    uint32_t get_group_id() const { return m_group_id; }
    uint32_t get_collective_id() const { return m_collective_id; }
    uint32_t get_world_size() const { return m_world_size; }
    int64_t get_axis() const { return m_axis; }

private:
    uint32_t m_group_id = 0;
    uint32_t m_collective_id = 0;
    uint32_t m_world_size = 1;
    int64_t m_axis = 0;
};

}  // namespace op
}  // namespace tp_gpu
}  // namespace ov
