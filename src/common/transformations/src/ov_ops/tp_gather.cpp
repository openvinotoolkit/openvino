// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/tp_gather.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {
namespace internal {

TPGather::TPGather(const ov::Output<ov::Node>& input,
                   uint32_t rank,
                   uint32_t world_size,
                   uint32_t group_id,
                   uint64_t collective_id,
                   int64_t axis)
    : TPCollectiveBase({input}, rank, world_size, group_id, collective_id),
      m_axis(axis) {
    constructor_validate_and_infer_types();
}

bool TPGather::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_TPGather_visit_attributes);
    TPCollectiveBase::visit_attributes(visitor);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void TPGather::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_TPGather_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, m_rank < m_world_size, "rank ", m_rank, " is outside of a group of ", m_world_size);

    auto pshape = get_input_partial_shape(0);
    // Only the root holds every slice; the others would otherwise each allocate the full tensor.
    if (m_rank == 0 && pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, pshape.rank().get_length() > 0, "a scalar has no axis to gather along");
        const auto axis = ov::util::try_normalize_axis(m_axis, pshape.rank(), *this);
        pshape[axis] *= static_cast<int64_t>(m_world_size);
    }
    set_output_type(0, get_input_element_type(0), pshape);
}

std::shared_ptr<ov::Node> TPGather::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_TPGather_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<TPGather>(new_args.at(0), m_rank, m_world_size, m_group_id, m_collective_id, m_axis);
}

}  // namespace internal
}  // namespace op
}  // namespace ov
