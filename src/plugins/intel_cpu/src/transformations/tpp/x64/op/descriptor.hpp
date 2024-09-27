// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "libxsmm_typedefs.h"
#include "openvino/core/except.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

class OpDescTPP {
public:
    // Note: zero arity represent equation arguments
    enum class ARITY {UNDEFINED, UNARY, BINARY, ZERO};
    OpDescTPP() = default;
    // Note: for zero arity op_type is interpreted as the argument index (op inputs and args have different order)
    OpDescTPP(ARITY arity, int arg_idx) : m_arity(arity), m_value{arg_idx}, m_flags{0} {
        OPENVINO_ASSERT(m_arity == ARITY::ZERO, "Only zero-arity op descs could be created directly");
    }
    explicit OpDescTPP(libxsmm_meltw_binary_type op_type, libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_BINARY_NONE) :
                m_arity{ARITY::BINARY}, m_value{op_type}, m_flags{flags} {}
    explicit OpDescTPP(libxsmm_meltw_unary_type op_type, libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_UNARY_NONE) :
                m_arity{ARITY::UNARY}, m_value{op_type}, m_flags{flags} {}
    operator libxsmm_meltw_binary_type() const {
        OPENVINO_ASSERT(m_arity == ARITY::BINARY, "Unsupported TPP OpDesc conversion");
        return static_cast<libxsmm_meltw_binary_type>(m_value);
    }
    operator libxsmm_meltw_unary_type() const {
        OPENVINO_ASSERT(m_arity == ARITY::UNARY, "Unsupported TPP OpDesc conversion");
        return static_cast<libxsmm_meltw_unary_type>(m_value);
    }
    explicit operator int() const {
        OPENVINO_ASSERT(m_arity == ARITY::ZERO, "Unsupported TPP OpDesc conversion");
        return m_value;
    }
    ARITY get_arity() const { return m_arity; }
    libxsmm_bitfield get_flags() const { return m_flags; }
    friend std::ostream& operator<<(std::ostream& os, const OpDescTPP& od);

private:
    const ARITY m_arity {ARITY::UNDEFINED};
    const int m_value {-1};
    const libxsmm_bitfield m_flags {0};
};

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
