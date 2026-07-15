// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_parallel/op/tp_reduce_scatter.hpp"

namespace ov {
namespace op {
namespace tp {

TPReduceScatter::TPReduceScatter(const Output<Node>& data,
                                 uint32_t group_id,
                                 uint32_t collective_id,
                                 uint32_t world_size,
                                 int64_t axis,
                                 const std::string& reduce_kind)
    : Op({data}),
      m_group_id(group_id),
      m_collective_id(collective_id),
      m_world_size(world_size),
      m_axis(axis),
      m_reduce_kind(reduce_kind) {
    constructor_validate_and_infer_types();
}

bool TPReduceScatter::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("group_id", m_group_id);
    visitor.on_attribute("collective_id", m_collective_id);
    visitor.on_attribute("world_size", m_world_size);
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("reduce_kind", m_reduce_kind);
    return true;
}

void TPReduceScatter::validate_and_infer_types() {
    auto pshape = get_input_partial_shape(0);
    if (pshape.rank().is_static()) {
        auto rank = pshape.rank().get_length();
        auto norm_axis = m_axis >= 0 ? m_axis : m_axis + rank;
        if (pshape[norm_axis].is_static()) {
            auto dim = pshape[norm_axis].get_length();
            OPENVINO_ASSERT(dim % m_world_size == 0,
                            "[TPReduceScatter] Dimension ", dim, " at axis ", m_axis,
                            " not divisible by world_size ", m_world_size);
            pshape[norm_axis] = dim / m_world_size;
        } else {
            pshape[norm_axis] = Dimension::dynamic();
        }
    }
    set_output_type(0, get_input_element_type(0), pshape);
}

std::shared_ptr<Node> TPReduceScatter::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<TPReduceScatter>(new_args.at(0), m_group_id, m_collective_id,
                                             m_world_size, m_axis, m_reduce_kind);
}

}  // namespace tp
}  // namespace op
}  // namespace ov
