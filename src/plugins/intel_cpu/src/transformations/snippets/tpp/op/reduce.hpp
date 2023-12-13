// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "modifiers.hpp"
#include "eltwise.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"


#include "libxsmm_typedefs.h"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

// Note: Reduce ops are implemented as UnaryEltwise in libxsmm, so we inherit this properties here
// Also note that UnaryEltwiseTPP is a modifier, so it won't trigger any flase positive matches in the pipeline
class ReduceMax : public UnaryEltwiseTPP, public ov::op::v1::ReduceMax {
public:
    OPENVINO_OP("ReduceMax", "TppOpset", ov::op::v1::ReduceMax);
    ReduceMax(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims = false);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
private:
    libxsmm_meltw_binary_type m_op_type;
};

class ReduceSum : public UnaryEltwiseTPP, public ov::op::v1::ReduceSum {
public:
    OPENVINO_OP("ReduceSum", "TppOpset", ov::op::v1::ReduceSum);
    ReduceSum(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims = false);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
private:
    libxsmm_meltw_binary_type m_op_type;
};

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
