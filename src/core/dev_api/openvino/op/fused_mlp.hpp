// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
/// \note FusedMLP op class is under development and subject to change
///
/// \brief Operator performing nothing
/// \ingroup ov_ops_cpp_api
class OPENVINO_API FusedMLP : public ov::op::Op {
public:
    OPENVINO_OP("FusedMLP");

    FusedMLP() = default;

    FusedMLP(const ov::Output<Node>& x,
             const ov::Output<Node>& w_gate,
             const ov::Output<Node>& w_up,
             const ov::Output<Node>& w_down);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
