// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

#define GENERAL_AUX_METHODS(OP, OP_TYPE, ...) \
 std::shared_ptr<Node> OP::clone_with_new_inputs(const OutputVector& new_args) const {\
    check_new_args_count(this, new_args);\
    const auto& new_op = std::make_shared<OP>(__VA_ARGS__);\
    new_op->clone_memory_acess_ports(*this);\
    return new_op;\
} \
 bool OP::visit_attributes(AttributeVisitor& visitor) {\
    return OP_TYPE::visit_attributes(visitor);\
}

#define BINARY_AUX_METHODS(BINARY_OP) GENERAL_AUX_METHODS(BINARY_OP, BinaryEltwiseTPP, new_args.at(0), new_args.at(1), this->get_autob())
#define UNARY_AUX_METHODS(UNARY_OP) GENERAL_AUX_METHODS(UNARY_OP, UnaryEltwiseTPP, new_args.at(0))

bool EltwiseTPP::is_supported(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v1::Add>(node) ||
           ov::is_type<ov::op::v1::Subtract>(node) ||
           ov::is_type<ov::op::v1::Multiply>(node) ||
           ov::is_type<ov::op::v1::Divide>(node);
}

bool EltwiseTPP::visit_attributes(AttributeVisitor& visitor) {
    std::string modifier{"TPP"};
    visitor.on_attribute("modifier", modifier);
    return MemoryAccess::visit_attributes(visitor);
}

BinaryEltwiseTPP::BinaryEltwiseTPP(libxsmm_meltw_binary_type op_type) : EltwiseTPP(), m_op_type(op_type) {
    // Initialize input/output ports as memory access ports
    ctor_initialize(std::set<size_t>{0, 1}, std::set<size_t>{0});
}

UnaryEltwiseTPP::UnaryEltwiseTPP(libxsmm_meltw_unary_type op_type) : EltwiseTPP(), m_op_type(op_type) {
    ctor_initialize(std::set<size_t>{0}, std::set<size_t>{0});
}

Add::Add(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
: BinaryEltwiseTPP(LIBXSMM_MELTW_TYPE_BINARY_ADD), ov::op::v1::Add(arg0, arg1, auto_broadcast) {
}

BINARY_AUX_METHODS(Add)

Subtract::Subtract(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
        : BinaryEltwiseTPP(LIBXSMM_MELTW_TYPE_BINARY_SUB), ov::op::v1::Subtract(arg0, arg1, auto_broadcast) {
}

BINARY_AUX_METHODS(Subtract)

Multiply::Multiply(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
        : BinaryEltwiseTPP(LIBXSMM_MELTW_TYPE_BINARY_MUL), ov::op::v1::Multiply(arg0, arg1, auto_broadcast) {
}

BINARY_AUX_METHODS(Multiply)

Divide::Divide(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
        : BinaryEltwiseTPP(LIBXSMM_MELTW_TYPE_BINARY_DIV), ov::op::v1::Divide(arg0, arg1, auto_broadcast) {
}

BINARY_AUX_METHODS(Divide)

Exp::Exp(const Output<Node>& arg0) : UnaryEltwiseTPP(LIBXSMM_MELTW_TYPE_UNARY_EXP), ov::op::v0::Exp(arg0) {
}

UNARY_AUX_METHODS(Exp)

Relu::Relu(const Output<Node>& arg0) : UnaryEltwiseTPP(LIBXSMM_MELTW_TYPE_UNARY_RELU), ov::op::v0::Relu(arg0) {
}

UNARY_AUX_METHODS(Relu)

Reciprocal::Reciprocal(const Output<Node>& arg) :
    UnaryEltwiseTPP(LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL), ov::snippets::op::PowerStatic(arg, -1.f) {
}

UNARY_AUX_METHODS(Reciprocal)

Square::Square(const Output<Node>& arg) :
    UnaryEltwiseTPP(LIBXSMM_MELTW_TYPE_UNARY_X2), ov::snippets::op::PowerStatic(arg, 2.f) {
}

UNARY_AUX_METHODS(Square)

SquareRoot::SquareRoot(const Output<Node>& arg) :
    UnaryEltwiseTPP(LIBXSMM_MELTW_TYPE_UNARY_SQRT), ov::snippets::op::PowerStatic(arg, 0.5f) {
}

UNARY_AUX_METHODS(SquareRoot)

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
