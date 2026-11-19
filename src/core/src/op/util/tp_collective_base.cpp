// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/tp_collective_base.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace util {

TPCollectiveBase::TPCollectiveBase(const OutputVector& arguments,
                                   uint32_t rank,
                                   uint32_t world_size,
                                   uint32_t group_id,
                                   uint64_t collective_id)
    : Op(arguments),
      m_rank(rank),
      m_world_size(world_size),
      m_group_id(group_id),
      m_collective_id(collective_id) {}

bool TPCollectiveBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_TPCollectiveBase_visit_attributes);
    visitor.on_attribute("rank", m_rank);
    visitor.on_attribute("world_size", m_world_size);
    visitor.on_attribute("group_id", m_group_id);
    visitor.on_attribute("collective_id", m_collective_id);
    return true;
}

}  // namespace util
}  // namespace op
}  // namespace ov
