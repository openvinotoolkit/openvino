// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/tp_all_reduce.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace internal {

TPAllReduce::TPAllReduce(const ov::Output<ov::Node>& input,
                         uint32_t rank,
                         uint32_t world_size,
                         uint32_t group_id,
                         uint64_t collective_id,
                         ReduceOp reduce_op)
    : TPCollectiveBase({input}, rank, world_size, group_id, collective_id),
      m_reduce_op(reduce_op) {
    constructor_validate_and_infer_types();
}

bool TPAllReduce::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_TPAllReduce_visit_attributes);
    TPCollectiveBase::visit_attributes(visitor);
    visitor.on_attribute("reduce_op", m_reduce_op);
    return true;
}

void TPAllReduce::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_TPAllReduce_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, m_rank < m_world_size, "rank ", m_rank, " is outside of a group of ", m_world_size);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> TPAllReduce::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_TPAllReduce_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<TPAllReduce>(new_args.at(0),
                                         m_rank,
                                         m_world_size,
                                         m_group_id,
                                         m_collective_id,
                                         m_reduce_op);
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
