// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp_gpu/op/tp_broadcast.hpp"

namespace ov {
namespace tp_gpu {
namespace op {

TPBroadcast::TPBroadcast(const Output<Node>& data,
                         uint32_t group_id,
                         uint32_t collective_id,
                         uint32_t world_size,
                         uint32_t root_rank)
    : Op({data}),
      m_group_id(group_id),
      m_collective_id(collective_id),
      m_world_size(world_size),
      m_root_rank(root_rank) {
    constructor_validate_and_infer_types();
}

bool TPBroadcast::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("group_id", m_group_id);
    visitor.on_attribute("collective_id", m_collective_id);
    visitor.on_attribute("world_size", m_world_size);
    visitor.on_attribute("root_rank", m_root_rank);
    return true;
}

void TPBroadcast::validate_and_infer_types() {
    // Broadcast: output shape == input shape, same element type
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> TPBroadcast::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<TPBroadcast>(new_args.at(0), m_group_id, m_collective_id, m_world_size, m_root_rank);
}

}  // namespace op
}  // namespace tp_gpu
}  // namespace ov
