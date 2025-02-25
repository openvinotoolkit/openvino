// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"

namespace ov::intel_cpu::tpp::op {

#define GENERAL_AUX_METHODS(OP, OP_TYPE, ...)                                             \
    std::shared_ptr<Node> OP::clone_with_new_inputs(const OutputVector& new_args) const { \
        check_new_args_count(this, new_args);                                             \
        const auto& new_op = std::make_shared<OP>(__VA_ARGS__);                           \
        new_op->clone_memory_access_ports(*this);                                         \
        return new_op;                                                                    \
    }                                                                                     \
    bool OP::visit_attributes(AttributeVisitor& visitor) { return OP_TYPE::visit_attributes(visitor); }

// Note: Unary Ops don't require broadcasting flags update => no need to override validate_and_infer_types
#define BINARY_AUX_METHODS(BINARY_OP, OV_OP)                                                            \
    GENERAL_AUX_METHODS(BINARY_OP, BinaryEltwiseTPP, new_args.at(0), new_args.at(1), this->get_autob()) \
    void BINARY_OP::validate_and_infer_types() {                                                        \
        OV_OP::validate_and_infer_types();                                                              \
        m_flags = get_broadcasting_flags(get_input_partial_shape(0), get_input_partial_shape(1));       \
    }

#define UNARY_AUX_METHODS(UNARY_OP) GENERAL_AUX_METHODS(UNARY_OP, UnaryEltwiseTPP, new_args.at(0))

bool EltwiseTPP::is_supported(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type_any_of<ov::op::v1::Add, ov::op::v1::Subtract, ov::op::v1::Multiply, ov::op::v1::Divide>(node);
}

bool EltwiseTPP::visit_attributes(AttributeVisitor& visitor) {
    TensorProcessingPrimitive::visit_attributes(visitor);
    return MemoryAccess::visit_attributes(visitor);
}

BinaryEltwiseTPP::BinaryEltwiseTPP(libxsmm_meltw_binary_type op_type) : EltwiseTPP(), m_op_type(op_type) {
    // Initialize input/output ports as memory access ports
    ctor_initialize(std::set<size_t>{0, 1}, std::set<size_t>{0});
}

libxsmm_bitfield BinaryEltwiseTPP::get_broadcasting_flags(const ov::PartialShape& pshape_0,
                                                          const ov::PartialShape& pshape_1) {
    return get_broadcasting_flags(snippets::utils::pshape_to_vdims(pshape_0),
                                  snippets::utils::pshape_to_vdims(pshape_1));
}

libxsmm_bitfield BinaryEltwiseTPP::get_broadcasting_flags(const snippets::VectorDims& shape_0,
                                                          const snippets::VectorDims& shape_1) {
    auto get_subshape = [](const snippets::VectorDims& shape) {
        snippets::VectorDims subshape(2, 1);
        for (size_t i = 0; i < std::min(subshape.size(), shape.size()); i++) {
            subshape[subshape.size() - 1 - i] = shape[shape.size() - 1 - i];
        }
        return subshape;
    };
    snippets::VectorDims subshape_0 = get_subshape(shape_0);
    snippets::VectorDims subshape_1 = get_subshape(shape_1);

    if (snippets::utils::is_dynamic_vdims(subshape_0) || snippets::utils::is_dynamic_vdims(subshape_1))
        return LIBXSMM_MELTW_FLAG_BINARY_NONE;
    if (subshape_0 == subshape_1) {
        return LIBXSMM_MELTW_FLAG_BINARY_NONE;
    } else if (ov::shape_size(subshape_0) == 1) {
        return LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0;
    } else if (ov::shape_size(subshape_1) == 1) {
        return LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
    } else {
        libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
        if (subshape_0[0] != subshape_1[0]) {
            if (subshape_0[0] == 1) {
                flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
            } else if (subshape_1[0] == 1) {
                flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
            } else {
                OPENVINO_THROW("Unsupported subshape combination: dim 0");
            }
        }
        if (subshape_0[1] != subshape_1[1]) {
            if (subshape_0[1] == 1) {
                flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0;
            } else if (subshape_1[1] == 1) {
                flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1;
            } else {
                OPENVINO_THROW("Unsupported subshape combination: dim 1");
            }
        }
        return flags;
    }
}

UnaryEltwiseTPP::UnaryEltwiseTPP(libxsmm_meltw_unary_type op_type) : EltwiseTPP(), m_op_type(op_type) {
    ctor_initialize(std::set<size_t>{0}, std::set<size_t>{0});
}

Add::Add(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryEltwiseTPP(LIBXSMM_MELTW_TYPE_BINARY_ADD),
      ov::op::v1::Add(arg0, arg1, auto_broadcast) {
    m_flags = get_broadcasting_flags(arg0.get_partial_shape(), arg1.get_partial_shape());
}

BINARY_AUX_METHODS(Add, ov::op::v1::Add)

Subtract::Subtract(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryEltwiseTPP(LIBXSMM_MELTW_TYPE_BINARY_SUB),
      ov::op::v1::Subtract(arg0, arg1, auto_broadcast) {
    m_flags = get_broadcasting_flags(arg0.get_partial_shape(), arg1.get_partial_shape());
}

BINARY_AUX_METHODS(Subtract, ov::op::v1::Subtract)

Multiply::Multiply(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryEltwiseTPP(LIBXSMM_MELTW_TYPE_BINARY_MUL),
      ov::op::v1::Multiply(arg0, arg1, auto_broadcast) {
    m_flags = get_broadcasting_flags(arg0.get_partial_shape(), arg1.get_partial_shape());
}

BINARY_AUX_METHODS(Multiply, ov::op::v1::Multiply)

Divide::Divide(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryEltwiseTPP(LIBXSMM_MELTW_TYPE_BINARY_DIV),
      ov::op::v1::Divide(arg0, arg1, auto_broadcast) {
    m_flags = get_broadcasting_flags(arg0.get_partial_shape(), arg1.get_partial_shape());
}

BINARY_AUX_METHODS(Divide, ov::op::v1::Divide)

Exp::Exp(const Output<Node>& arg0) : UnaryEltwiseTPP(LIBXSMM_MELTW_TYPE_UNARY_EXP), ov::op::v0::Exp(arg0) {}

UNARY_AUX_METHODS(Exp)

Relu::Relu(const Output<Node>& arg0) : UnaryEltwiseTPP(LIBXSMM_MELTW_TYPE_UNARY_RELU), ov::op::v0::Relu(arg0) {}

UNARY_AUX_METHODS(Relu)

Reciprocal::Reciprocal(const Output<Node>& arg)
    : UnaryEltwiseTPP(LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL),
      ov::snippets::op::PowerStatic(arg, -1.f) {}

UNARY_AUX_METHODS(Reciprocal)

Square::Square(const Output<Node>& arg)
    : UnaryEltwiseTPP(LIBXSMM_MELTW_TYPE_UNARY_X2),
      ov::snippets::op::PowerStatic(arg, 2.f) {}

UNARY_AUX_METHODS(Square)

SquareRoot::SquareRoot(const Output<Node>& arg)
    : UnaryEltwiseTPP(LIBXSMM_MELTW_TYPE_UNARY_SQRT),
      ov::snippets::op::PowerStatic(arg, 0.5f) {}

UNARY_AUX_METHODS(SquareRoot)

}  // namespace ov::intel_cpu::tpp::op
