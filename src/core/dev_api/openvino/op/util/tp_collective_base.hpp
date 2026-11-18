// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {

/// \brief Base of the tensor-parallel collectives.
///
/// Carries the collective group a call belongs to, this rank's place in it and the collective's id; nothing
/// about how the call is carried out. The device plugin compiling the op provides the communication.
///
/// The id is set by whoever inserts the collectives: unique within the model, the same on every rank for
/// the same collective, and stable for the same model. Device plugins compare ids across the ranks and
/// may not assume they are dense or ordered.
class OPENVINO_API TPCollectiveBase : public Op {
public:
    // Own RTTI, so that is_type<TPCollectiveBase> matches the collectives only.
    OPENVINO_OP("TPCollectiveBase", "util");

    TPCollectiveBase() = default;

    uint32_t get_rank() const {
        return m_rank;
    }
    uint32_t get_world_size() const {
        return m_world_size;
    }
    uint32_t get_group_id() const {
        return m_group_id;
    }
    uint64_t get_collective_id() const {
        return m_collective_id;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    TPCollectiveBase(const OutputVector& arguments,
                     uint32_t rank,
                     uint32_t world_size,
                     uint32_t group_id,
                     uint64_t collective_id);

    uint32_t m_rank = 0;
    uint32_t m_world_size = 1;
    uint32_t m_group_id = 0;
    uint64_t m_collective_id = 0;
};

}  // namespace util
}  // namespace op
}  // namespace ov
