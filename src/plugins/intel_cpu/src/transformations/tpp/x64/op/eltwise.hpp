// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "modifiers.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/relu.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/utils/utils.hpp"

#include "descriptor.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {
using AutoBroadcastSpec = ov::op::AutoBroadcastSpec;
using AutoBroadcastType = ov::op::AutoBroadcastType;

class EltwiseTPP : public modifier::TensorProcessingPrimitive {
public:
    static bool is_supported(const std::shared_ptr<ov::Node>& node);
    bool visit_attributes(AttributeVisitor& visitor);
    virtual OpDescTPP get_op_desc() const  = 0;
};

class BinaryEltwiseTPP : public EltwiseTPP {
public:
    BinaryEltwiseTPP(libxsmm_meltw_binary_type op_type);
    OpDescTPP get_op_desc() const override { return OpDescTPP(m_op_type, m_flags); }

protected:
    static libxsmm_bitfield get_broadcasting_flags(const ov::PartialShape& pshape_0, const ov::PartialShape& pshape_1);
    static libxsmm_bitfield get_broadcasting_flags(const snippets::VectorDims& pshape_0, const snippets::VectorDims& pshape_1);
    libxsmm_bitfield m_flags;
    libxsmm_meltw_binary_type m_op_type;
};

class UnaryEltwiseTPP : public EltwiseTPP {
public:
    UnaryEltwiseTPP(libxsmm_meltw_unary_type op_type);
    OpDescTPP get_op_desc() const override { return OpDescTPP(m_op_type); }
private:
    libxsmm_meltw_unary_type m_op_type;
};

class Add : public BinaryEltwiseTPP, public ov::op::v1::Add {
public:
    OPENVINO_OP("Add", "TppOpset", ov::op::v1::Add);
    Add(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

class Subtract : public BinaryEltwiseTPP, public ov::op::v1::Subtract {
public:
    OPENVINO_OP("Subtract", "TppOpset", ov::op::v1::Subtract);
    Subtract(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

class Multiply : public BinaryEltwiseTPP, public ov::op::v1::Multiply {
public:
    OPENVINO_OP("Multiply", "TppOpset", ov::op::v1::Multiply);
    Multiply(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

class Divide : public BinaryEltwiseTPP, public ov::op::v1::Divide {
public:
    OPENVINO_OP("Divide", "TppOpset", ov::op::v1::Divide);
    Divide(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast);
    void validate_and_infer_types() override;
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

class Relu : public UnaryEltwiseTPP, public ov::op::v0::Relu {
public:
    OPENVINO_OP("Relu", "TppOpset", ov::op::v0::Relu);
    Relu(const Output<Node>& arg);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

class Reciprocal : public UnaryEltwiseTPP, public ov::snippets::op::PowerStatic {
public:
    OPENVINO_OP("Reciprocal", "TppOpset", snippets::op::PowerStatic);
    Reciprocal(const Output<Node>& arg);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};


class Square : public UnaryEltwiseTPP, public ov::snippets::op::PowerStatic {
public:
    OPENVINO_OP("Square", "TppOpset", snippets::op::PowerStatic);
    Square(const Output<Node>& arg);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

class SquareRoot : public UnaryEltwiseTPP, public ov::snippets::op::PowerStatic {
public:
    OPENVINO_OP("SquareRoot", "TppOpset", snippets::op::PowerStatic);
    SquareRoot(const Output<Node>& arg);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
