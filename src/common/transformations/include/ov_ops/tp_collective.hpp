// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <ostream>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

enum class ReduceOp { SUM, MAX, MIN };

/// \brief Base of the tensor-parallel collectives.
///
/// Carries the collective group a call belongs to and this rank's place in it; nothing about how the
/// call is carried out. The device plugin compiling the op provides the communication.
class TRANSFORMATIONS_API TPCollectiveBase : public ov::op::Op {
public:
    // Own RTTI, so that is_type<TPCollectiveBase> matches the collectives only.
    OPENVINO_OP("TPCollectiveBase", "util");

    TPCollectiveBase() = default;

    int get_rank() const {
        return m_rank;
    }
    int get_world_size() const {
        return m_world_size;
    }
    int get_group_id() const {
        return m_group_id;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

protected:
    TPCollectiveBase(const ov::OutputVector& arguments, int rank, int world_size, int group_id);

    int m_rank = 0;
    int m_world_size = 1;
    int m_group_id = 0;
};

/// \brief Element-wise reduction of the inputs of all ranks of the group; every rank gets the result.
class TRANSFORMATIONS_API TPAllReduce : public TPCollectiveBase {
public:
    OPENVINO_OP("TPAllReduce", "ie_internal_opset", TPCollectiveBase);

    TPAllReduce() = default;
    TPAllReduce(const ov::Output<ov::Node>& input, int rank, int world_size, int group_id, ReduceOp reduce_op);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ReduceOp get_reduce_op() const {
        return m_reduce_op;
    }

private:
    ReduceOp m_reduce_op = ReduceOp::SUM;
};

/// \brief Concatenation of the inputs of all ranks along `axis`, in rank order, on rank 0.
///
/// Only rank 0's output holds every slice; the other ranks' output has their input's shape and unspecified
/// contents.
class TRANSFORMATIONS_API TPGather : public TPCollectiveBase {
public:
    OPENVINO_OP("TPGather", "ie_internal_opset", TPCollectiveBase);

    TPGather() = default;
    TPGather(const ov::Output<ov::Node>& input, int rank, int world_size, int group_id, int64_t axis = -1);

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

std::ostream& operator<<(std::ostream& s, const ov::op::internal::ReduceOp& reduce_op);

template <>
class OPENVINO_API AttributeAdapter<ov::op::internal::ReduceOp>
    : public EnumAttributeAdapterBase<ov::op::internal::ReduceOp> {
public:
    AttributeAdapter(ov::op::internal::ReduceOp& value) : EnumAttributeAdapterBase<ov::op::internal::ReduceOp>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::internal::ReduceOp>");
};

}  // namespace ov
