// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <ostream>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/op/util/tp_collective_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

enum class ReduceOp { SUM, MAX, MIN };

/// \brief Element-wise reduction of the inputs of all ranks of the group; every rank gets the result.
class TRANSFORMATIONS_API TPAllReduce : public ov::op::util::TPCollectiveBase {
public:
    OPENVINO_OP("TPAllReduce", "ie_internal_opset", ov::op::util::TPCollectiveBase);

    TPAllReduce() = default;
    TPAllReduce(const ov::Output<ov::Node>& input,
                uint32_t rank,
                uint32_t world_size,
                uint32_t group_id,
                uint64_t collective_id,
                ReduceOp reduce_op);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ReduceOp get_reduce_op() const {
        return m_reduce_op;
    }

private:
    ReduceOp m_reduce_op = ReduceOp::SUM;
};

}  // namespace internal
}  // namespace op

std::ostream& operator<<(std::ostream& s, const ov::op::internal::ReduceOp& reduce_op);

template <>
class OPENVINO_API AttributeAdapter<ov::op::internal::ReduceOp>
    : public EnumAttributeAdapterBase<ov::op::internal::ReduceOp> {
public:
    AttributeAdapter(ov::op::internal::ReduceOp& value) : EnumAttributeAdapterBase<ov::op::internal::ReduceOp>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::internal::ReduceOp>");
};

}  // namespace ov
