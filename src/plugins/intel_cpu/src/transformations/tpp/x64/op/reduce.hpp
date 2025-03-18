// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "eltwise.hpp"
#include "libxsmm_typedefs.h"
#include "snippets/op/reduce.hpp"
#include "transformations/tpp/common/op/modifiers.hpp"

namespace ov::intel_cpu::tpp::op {

// Note: Reduce ops are implemented as UnaryEltwise in libxsmm, so we inherit this properties here
// Also note that UnaryEltwiseTPP is a modifier, so it won't trigger any flase positive matches in the pipeline
class ReduceMax : public UnaryEltwiseTPP, public ov::snippets::op::ReduceMax {
public:
    OPENVINO_OP("ReduceMax", "TppOpset", ov::snippets::op::ReduceMax);
    ReduceMax(const Output<Node>& arg, size_t axis);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    libxsmm_meltw_binary_type m_op_type;
};

class ReduceSum : public UnaryEltwiseTPP, public ov::snippets::op::ReduceSum {
public:
    OPENVINO_OP("ReduceSum", "TppOpset", ov::snippets::op::ReduceSum);
    ReduceSum(const Output<Node>& arg, size_t axis);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    libxsmm_meltw_binary_type m_op_type;
};

}  // namespace ov::intel_cpu::tpp::op
