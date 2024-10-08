// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "descriptor.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

std::ostream& operator<<(std::ostream& os, const OpDescTPP& od) {
    switch (od.m_arity) {
        case OpDescTPP::ARITY::ZERO:
            os << "ARG#" << static_cast<int>(od.m_value);
            break;
        case OpDescTPP::ARITY::UNARY:
            switch (static_cast<libxsmm_meltw_unary_type>(od.m_value)) {
                case LIBXSMM_MELTW_TYPE_UNARY_EXP:
                    os << "EXP";
                    break;
                case LIBXSMM_MELTW_TYPE_UNARY_X2:
                    os << "SQ";
                    break;
                case LIBXSMM_MELTW_TYPE_UNARY_SQRT:
                    os << "SQRT";
                    break;
                case LIBXSMM_MELTW_TYPE_UNARY_RELU:
                    os << "RELU";
                    break;
                case LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL:
                    os << "RECIPROCAL";
                    break;
                case LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD:
                    os << "REDUCE_ADD";
                    break;
                case LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX:
                    os << "REDUCE_MAX";
                    break;
                default:
                    OPENVINO_THROW("Unsupported TPP Unary op type for serialization");
            }
            break;
        case OpDescTPP::ARITY::BINARY:
            switch (static_cast<libxsmm_meltw_binary_type>(od.m_value)) {
                case LIBXSMM_MELTW_TYPE_BINARY_ADD:
                    os << "ADD";
                    break;
                case LIBXSMM_MELTW_TYPE_BINARY_SUB:
                    os << "SUB";
                    break;
                case LIBXSMM_MELTW_TYPE_BINARY_MUL:
                    os << "MUL";
                    break;
                case LIBXSMM_MELTW_TYPE_BINARY_DIV:
                    os << "DIV";
                    break;
                default:
                    OPENVINO_THROW("Unsupported TPP Binary op type for serialization");
            }
            break;
        case OpDescTPP::ARITY::UNDEFINED:
            os << "Undefined";
            break;
        default:
            OPENVINO_THROW("Unhandled ARITY");
    }
    return os;
}
} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
