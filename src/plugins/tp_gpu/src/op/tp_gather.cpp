// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp_gpu/op/tp_gather.hpp"

namespace ov {
namespace tp_gpu {
namespace op {

TPGather::TPGather(const Output<Node>& data,
                   uint32_t group_id,
                   uint32_t collective_id,
                   uint32_t rank,
                   uint32_t world_size,
                   int64_t axis)
    : Op({data}),
      m_group_id(group_id),
      m_collective_id(collective_id),
      m_rank(rank),
      m_world_size(world_size),
      m_axis(axis) {
    constructor_validate_and_infer_types();
}

bool TPGather::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("group_id", m_group_id);
    visitor.on_attribute("collective_id", m_collective_id);
    visitor.on_attribute("rank", m_rank);
    visitor.on_attribute("world_size", m_world_size);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void TPGather::validate_and_infer_types() {
    auto pshape = get_input_partial_shape(0);

    // Only the root ends up holding every slice; the others keep their own,
    // which is what stops this from allocating a full-vocabulary tensor on
    // every rank -- at prompt length that would be over a hundred megabytes
    // each, written by no one.
    if (m_rank == 0 && pshape.rank().is_static()) {
        const auto rank_len = pshape.rank().get_length();
        const auto norm_axis = m_axis >= 0 ? m_axis : m_axis + rank_len;
        OPENVINO_ASSERT(norm_axis >= 0 && norm_axis < rank_len,
                        "[TP_GPU] TPGather axis ", m_axis, " is out of range for a rank-",
                        rank_len, " input");
        if (pshape[norm_axis].is_static()) {
            pshape[norm_axis] = pshape[norm_axis].get_length() * static_cast<int64_t>(m_world_size);
        } else {
            pshape[norm_axis] = Dimension::dynamic();
        }
    }

    set_output_type(0, get_input_element_type(0), pshape);
}

std::shared_ptr<Node> TPGather::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<TPGather>(new_args.at(0),
                                      m_group_id,
                                      m_collective_id,
                                      m_rank,
                                      m_world_size,
                                      m_axis);
}

}  // namespace op
}  // namespace tp_gpu
}  // namespace ov
