// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

ReduceMax::ReduceMax(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims) 
    : UnaryEltwiseTPP(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX), ov::op::v1::ReduceMax(arg, reduction_axes, keep_dims) {
}

std::shared_ptr<Node> ReduceMax::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    const auto& new_op = std::make_shared<ReduceMax>(new_args.at(0));
    new_op->clone_memory_acess_ports(*this);
    return new_op;
}

bool ReduceMax::visit_attributes(AttributeVisitor& visitor) {
    ArithmeticReductionKeepDims::visit_attributes(visitor);
    return UnaryEltwiseTPP::visit_attributes(visitor);
}

ReduceSum::ReduceSum(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims) 
    : UnaryEltwiseTPP(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD), ov::op::v1::ReduceSum(arg, reduction_axes, keep_dims) {
}

std::shared_ptr<Node> ReduceSum::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    const auto& new_op = std::make_shared<ReduceSum>(new_args.at(0));
    new_op->clone_memory_acess_ports(*this);
    return new_op;
}

bool ReduceSum::visit_attributes(AttributeVisitor& visitor) {
    ArithmeticReductionKeepDims::visit_attributes(visitor);
    return UnaryEltwiseTPP::visit_attributes(visitor);
}

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
