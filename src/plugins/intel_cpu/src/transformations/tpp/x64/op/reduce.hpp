// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "descriptor.hpp"
#include "libxsmm_typedefs.h"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "snippets/op/reduce.hpp"
#include "transformations/tpp/common/op/modifiers.hpp"

namespace ov::intel_cpu::tpp::op {

class ReduceTPP : public modifier::TensorProcessingPrimitive {
public:
    bool visit_attributes(AttributeVisitor& visitor);
    [[nodiscard]] virtual OpDescTPP get_op_desc() const = 0;

protected:
    ReduceTPP(libxsmm_meltw_unary_type op_type);
    libxsmm_meltw_unary_type m_op_type;
};

class ReduceMax : public ReduceTPP, public ov::snippets::op::ReduceMax {
public:
    OPENVINO_OP("ReduceMax", "TppOpset", ov::snippets::op::ReduceMax);
    ReduceMax(const Output<Node>& arg, size_t axis);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    [[nodiscard]] OpDescTPP get_op_desc() const override {
        return OpDescTPP(m_op_type);
    }
};

class ReduceSum : public ReduceTPP, public ov::snippets::op::ReduceSum {
public:
    OPENVINO_OP("ReduceSum", "TppOpset", ov::snippets::op::ReduceSum);
    ReduceSum(const Output<Node>& arg, size_t axis);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    [[nodiscard]] OpDescTPP get_op_desc() const override {
        return OpDescTPP(m_op_type);
    }
};

}  // namespace ov::intel_cpu::tpp::op
