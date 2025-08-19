// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_emitters.hpp"

#include <common/utils.hpp>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "snippets/lowered/expression.hpp"
#include "utils/general_utils.h"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64 {

using jit_generator_t = ov::intel_cpu::riscv64::jit_generator_t;
using cpu_isa_t = ov::intel_cpu::riscv64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_nop_emitter::jit_nop_emitter(jit_generator_t* h, cpu_isa_t isa, [[maybe_unused]] const ExpressionPtr& expr)
    : riscv64::jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

jit_scalar_emitter::jit_scalar_emitter(jit_generator_t* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    const auto n = expr->get_node();
    const auto& precision = n->get_output_element_type(0);
    switch (precision) {
    case element::i32: {
        value = ov::as_type_ptr<ov::op::v0::Constant>(n)->cast_vector<int32_t>()[0];
        break;
    }
    case element::f32: {
        // For RISC-V, we'll store the float value as int32 bitcast
        const auto float_val = ov::as_type_ptr<ov::op::v0::Constant>(n)->cast_vector<float>()[0];
        value = *reinterpret_cast<const int32_t*>(&float_val);
        break;
    }
    default: {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support precision ", precision);
    }
    }
    // Store the value directly - no table needed for this simple implementation
}

void jit_scalar_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (host_isa_ == ov::intel_cpu::riscv64::gv) {
        emit_isa<ov::intel_cpu::riscv64::gv>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support isa ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_scalar_emitter::emit_isa([[maybe_unused]] const std::vector<size_t>& in,
                                  const std::vector<size_t>& out) const {
    // Get destination vector register
    Xbyak_riscv::VReg dst_vreg = Xbyak_riscv::VReg(out[0]);
    
    // For now, use t0 as a temporary register
    Xbyak_riscv::Reg tmp_gpr = Xbyak_riscv::t0;
    
    // Load scalar value directly into register
    h->uni_li(tmp_gpr, value);
    
    // Broadcast scalar to vector register using RISC-V Vector Extension
    // Set vector configuration for 32-bit elements
    h->vsetivli(Xbyak_riscv::zero, 4, Xbyak_riscv::SEW::e32, Xbyak_riscv::LMUL::m1);
    
    // Move scalar from GPR to vector register and broadcast
    h->vmv_v_x(dst_vreg, tmp_gpr);
}

}  // namespace ov::intel_cpu::riscv64