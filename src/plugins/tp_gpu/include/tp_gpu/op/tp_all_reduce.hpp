// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace tp_gpu {
namespace op {

/// \brief All-reduce collective: sums a tensor across every TP rank, in place.
class TPAllReduce : public ov::op::Op {
public:
    OPENVINO_OP("TPAllReduce", "tp_gpu", Op);

    TPAllReduce() = default;

    /// \param data           This rank's partial sum.
    /// \param collective_id  Which exchange this is; the ranks meet by it, so
    ///                       it has to be the same one on every rank.
    /// \param rank           This rank's index.
    /// \param world_size     How many ranks contribute.
    TPAllReduce(const Output<Node>& data,
                uint32_t collective_id,
                uint32_t rank,
                uint32_t world_size);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    uint32_t get_collective_id() const { return m_collective_id; }
    uint32_t get_rank() const { return m_rank; }
    uint32_t get_world_size() const { return m_world_size; }

private:
    uint32_t m_collective_id = 0;
    uint32_t m_rank = 0;
    uint32_t m_world_size = 1;
};

}  // namespace op
}  // namespace tp_gpu
}  // namespace ov
