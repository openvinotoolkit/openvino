// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_parallel/op/tp_all_gather.hpp"

namespace ov {
namespace op {
namespace tp {

TPAllGather::TPAllGather(const Output<Node>& data,
                         uint32_t group_id,
                         uint32_t collective_id,
                         uint32_t world_size,
                         int64_t axis)
    : Op({data}),
      m_group_id(group_id),
      m_collective_id(collective_id),
      m_world_size(world_size),
      m_axis(axis) {
    constructor_validate_and_infer_types();
}

bool TPAllGather::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("group_id", m_group_id);
    visitor.on_attribute("collective_id", m_collective_id);
    visitor.on_attribute("world_size", m_world_size);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void TPAllGather::validate_and_infer_types() {
    auto pshape = get_input_partial_shape(0);
    if (pshape.rank().is_static()) {
        auto rank = pshape.rank().get_length();
        auto norm_axis = m_axis >= 0 ? m_axis : m_axis + rank;
        if (pshape[norm_axis].is_static()) {
            pshape[norm_axis] = pshape[norm_axis].get_length() * m_world_size;
        } else {
            pshape[norm_axis] = Dimension::dynamic();
        }
    }
    set_output_type(0, get_input_element_type(0), pshape);
}

std::shared_ptr<Node> TPAllGather::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<TPAllGather>(new_args.at(0), m_group_id, m_collective_id, m_world_size, m_axis);
}

}  // namespace tp
}  // namespace op
}  // namespace ov
