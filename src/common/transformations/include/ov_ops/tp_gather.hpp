// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/op/util/tp_collective_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

/// \brief Concatenation of the inputs of all ranks along `axis`, in rank order, on rank 0.
///
/// Only rank 0's output holds every slice; the other ranks' output has their input's shape and unspecified
/// contents.
class TRANSFORMATIONS_API TPGather : public ov::op::util::TPCollectiveBase {
public:
    OPENVINO_OP("TPGather", "ie_internal_opset", ov::op::util::TPCollectiveBase);

    TPGather() = default;
    TPGather(const ov::Output<ov::Node>& input,
             uint32_t rank,
             uint32_t world_size,
             uint32_t group_id,
             uint64_t collective_id,
             int64_t axis = -1);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    int64_t get_axis() const {
        return m_axis;
    }

private:
    int64_t m_axis = -1;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
