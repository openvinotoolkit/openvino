// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "modifiers.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"

#include "libxsmm_typedefs.h"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {
using AutoBroadcastSpec = ov::op::AutoBroadcastSpec;
using AutoBroadcastType = ov::op::AutoBroadcastType;

class EltwiseTPP : public modifier::TensorProcessingPrimitive {
public:
    EltwiseTPP();
    static bool is_supported(const std::shared_ptr<ov::Node>& node);
    bool visit_attributes(AttributeVisitor& visitor);
};

class BinaryEltwiseTPP : public EltwiseTPP {
public:
    BinaryEltwiseTPP(libxsmm_meltw_binary_type op_type) : m_op_type(op_type) {}
    libxsmm_meltw_binary_type get_op_type() const { return m_op_type; }
private:
    libxsmm_meltw_binary_type m_op_type;
};

class UnaryEltwiseTPP : public EltwiseTPP {
public:
    UnaryEltwiseTPP(libxsmm_meltw_unary_type op_type) : m_op_type(op_type) {}
    libxsmm_meltw_unary_type get_op_type() const { return m_op_type; }
private:
    libxsmm_meltw_unary_type m_op_type;
};

class Add : public BinaryEltwiseTPP, public ov::op::v1::Add {
public:
    OPENVINO_OP("Add", "TppOpset", ov::op::v1::Add);
    Add(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

class Subtract : public BinaryEltwiseTPP, public ov::op::v1::Subtract {
public:
    OPENVINO_OP("Subtract", "TppOpset", ov::op::v1::Subtract);
    Subtract(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

class Multiply : public BinaryEltwiseTPP, public ov::op::v1::Multiply {
public:
    OPENVINO_OP("Multiply", "TppOpset", ov::op::v1::Multiply);
    Multiply(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

class Divide : public BinaryEltwiseTPP, public ov::op::v1::Divide {
public:
    OPENVINO_OP("Divide", "TppOpset", ov::op::v1::Divide);
    Divide(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

class Exp : public UnaryEltwiseTPP, public ov::op::v0::Exp {
public:
    OPENVINO_OP("Exp", "TppOpset", ov::op::v0::Exp);
    Exp(const Output<Node>& arg);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
