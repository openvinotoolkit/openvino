// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_parallel/op/tp_all_reduce.hpp"

namespace ov {
namespace op {
namespace tp {

TPAllReduce::TPAllReduce(const Output<Node>& data,
                         uint32_t group_id,
                         uint32_t collective_id,
                         uint32_t world_size,
                         const std::string& reduce_kind)
    : Op({data}),
      m_group_id(group_id),
      m_collective_id(collective_id),
      m_world_size(world_size),
      m_reduce_kind(reduce_kind) {
    constructor_validate_and_infer_types();
}

bool TPAllReduce::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("group_id", m_group_id);
    visitor.on_attribute("collective_id", m_collective_id);
    visitor.on_attribute("world_size", m_world_size);
    visitor.on_attribute("reduce_kind", m_reduce_kind);
    return true;
}

void TPAllReduce::validate_and_infer_types() {
    // AllReduce: output shape == input shape, same element type
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> TPAllReduce::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<TPAllReduce>(new_args.at(0), m_group_id, m_collective_id, m_world_size, m_reduce_kind);
}

}  // namespace tp
}  // namespace op
}  // namespace ov
