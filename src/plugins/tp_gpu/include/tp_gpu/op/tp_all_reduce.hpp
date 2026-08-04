// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace tp_gpu {
namespace op {

/// \brief All-reduce collective: sums (or reduces) a tensor across all TP ranks.
/// Output shape == input shape.
class TPAllReduce : public ov::op::Op {
public:
    OPENVINO_OP("TPAllReduce", "tp_gpu", Op);

    TPAllReduce() = default;

    /// \param data   The tensor to reduce
    /// \param group_id       Logical group identifier
    /// \param collective_id  Unique id for matching across ranks
    /// \param world_size     Number of participating ranks
    /// \param reduce_kind    "sum", "mean" (extensible)
    TPAllReduce(const Output<Node>& data,
                uint32_t group_id,
                uint32_t collective_id,
                uint32_t world_size,
                const std::string& reduce_kind = "sum");

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    uint32_t get_group_id() const { return m_group_id; }
    uint32_t get_collective_id() const { return m_collective_id; }
    uint32_t get_world_size() const { return m_world_size; }
    const std::string& get_reduce_kind() const { return m_reduce_kind; }

private:
    uint32_t m_group_id = 0;
    uint32_t m_collective_id = 0;
    uint32_t m_world_size = 1;
    std::string m_reduce_kind = "sum";
};

}  // namespace op
}  // namespace tp_gpu
}  // namespace ov
