// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/tp_collective.hpp"

namespace ov {
namespace op {
namespace internal {

TPCollectiveBase::TPCollectiveBase(const ov::OutputVector& arguments, int rank, int world_size, int group_id)
    : Op(arguments),
      m_rank(rank),
      m_world_size(world_size),
      m_group_id(group_id) {}

bool TPCollectiveBase::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("rank", m_rank);
    visitor.on_attribute("world_size", m_world_size);
    visitor.on_attribute("group_id", m_group_id);
    return true;
}

TPAllReduce::TPAllReduce(const ov::Output<ov::Node>& input,
                         int rank,
                         int world_size,
                         int group_id,
                         ReduceOp reduce_op)
    : TPCollectiveBase({input}, rank, world_size, group_id),
      m_reduce_op(reduce_op) {
    constructor_validate_and_infer_types();
}

bool TPAllReduce::visit_attributes(ov::AttributeVisitor& visitor) {
    TPCollectiveBase::visit_attributes(visitor);
    visitor.on_attribute("reduce_op", m_reduce_op);
    return true;
}

void TPAllReduce::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this,
                          m_world_size >= 1 && m_rank >= 0 && m_rank < m_world_size,
                          "rank ",
                          m_rank,
                          " is outside of a group of ",
                          m_world_size);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> TPAllReduce::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<TPAllReduce>(new_args.at(0), m_rank, m_world_size, m_group_id, m_reduce_op);
}

TPGather::TPGather(const ov::Output<ov::Node>& input, int rank, int world_size, int group_id, int64_t axis)
    : TPCollectiveBase({input}, rank, world_size, group_id),
      m_axis(axis) {
    constructor_validate_and_infer_types();
}

bool TPGather::visit_attributes(ov::AttributeVisitor& visitor) {
    TPCollectiveBase::visit_attributes(visitor);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void TPGather::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this,
                          m_world_size >= 1 && m_rank >= 0 && m_rank < m_world_size,
                          "rank ",
                          m_rank,
                          " is outside of a group of ",
                          m_world_size);

    auto pshape = get_input_partial_shape(0);
    // Only the root holds every slice; the others would otherwise each allocate the full tensor.
    if (m_rank == 0 && pshape.rank().is_static()) {
        const auto rank_len = pshape.rank().get_length();
        const auto norm_axis = m_axis >= 0 ? m_axis : m_axis + rank_len;
        NODE_VALIDATION_CHECK(this,
                              norm_axis >= 0 && norm_axis < rank_len,
                              "axis ",
                              m_axis,
                              " is out of range for a rank-",
                              rank_len,
                              " input");
        if (pshape[norm_axis].is_static()) {
            pshape[norm_axis] = pshape[norm_axis].get_length() * static_cast<int64_t>(m_world_size);
        } else {
            pshape[norm_axis] = ov::Dimension::dynamic();
        }
    }
    set_output_type(0, get_input_element_type(0), pshape);
}

std::shared_ptr<ov::Node> TPGather::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<TPGather>(new_args.at(0), m_rank, m_world_size, m_group_id, m_axis);
}

}  // namespace internal
}  // namespace op

std::ostream& operator<<(std::ostream& s, const ov::op::internal::ReduceOp& reduce_op) {
    return s << ov::as_string(reduce_op);
}

template <>
OPENVINO_API EnumNames<ov::op::internal::ReduceOp>& EnumNames<ov::op::internal::ReduceOp>::get() {
    static auto enum_names = EnumNames<ov::op::internal::ReduceOp>("ov::op::internal::ReduceOp",
                                                                   {
                                                                       {"sum", ov::op::internal::ReduceOp::SUM},
                                                                       {"max", ov::op::internal::ReduceOp::MAX},
                                                                       {"min", ov::op::internal::ReduceOp::MIN},
                                                                   });
    return enum_names;
}

}  // namespace ov
