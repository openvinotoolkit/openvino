// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_emitters.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "emitters/utils.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/broadcastmove.hpp"
#include "utils/general_utils.h"
#include "xbyak_riscv/xbyak_riscv.hpp"
#include "xbyak_riscv/xbyak_riscv_csr.hpp"

namespace ov::intel_cpu::riscv64 {

using jit_generator_t = ov::intel_cpu::riscv64::jit_generator_t;
using cpu_isa_t = ov::intel_cpu::riscv64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_nop_emitter::jit_nop_emitter(jit_generator_t* h, cpu_isa_t isa, [[maybe_unused]] const ExpressionPtr& expr)
    : riscv64::jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

jit_broadcast_move_emitter::jit_broadcast_move_emitter(jit_generator_t* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    const auto n = expr->get_node();
    OV_CPU_JIT_EMITTER_ASSERT(ov::as_type_ptr<snippets::op::BroadcastMove>(n) != nullptr,
                              "Expects BroadcastMove expression");
    const auto element_type = n->get_input_element_type(0);
    OV_CPU_JIT_EMITTER_ASSERT(any_of(element_type.size(), 1U, 2U, 4U), "Unsupported element type: ", element_type);
    byte_size = element_type.size();
}

void jit_broadcast_move_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (host_isa_ == ov::intel_cpu::riscv64::gv) {
        emit_isa<ov::intel_cpu::riscv64::gv>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support isa ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_broadcast_move_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    auto src_vreg = Xbyak_riscv::VReg(in[0]);
    auto dst_vreg = Xbyak_riscv::VReg(out[0]);
    // Due to the fact that InsertBroadcastMove may happen after register allocation in case of dynamic shapes, we can
    // end up in a situation where source and destination registers are the same
    const bool in_place = src_vreg.getIdx() == dst_vreg.getIdx();

    switch (byte_size) {
    case 1:
        h->vsetivli(Xbyak_riscv::zero, 4, Xbyak_riscv::SEW::e8, Xbyak_riscv::LMUL::m1);
        break;
    case 2:
        h->vsetivli(Xbyak_riscv::zero, 4, Xbyak_riscv::SEW::e16, Xbyak_riscv::LMUL::m1);
        break;
    case 4:
        h->vsetivli(Xbyak_riscv::zero, 4, Xbyak_riscv::SEW::e32, Xbyak_riscv::LMUL::m1);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported data size ", byte_size);
    }

    if (in_place) {
        OV_CPU_JIT_EMITTER_ASSERT(!aux_vec_idxs.empty(), "BroadcastMove requires an auxiliary vector register");
        const auto tmp_vreg = Xbyak_riscv::VReg(aux_vec_idxs.back());
        h->vrgather_vi(tmp_vreg, src_vreg, 0);
        h->vmv_v_v(dst_vreg, tmp_vreg);
    } else {
        h->vrgather_vi(dst_vreg, src_vreg, 0);
    }
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
        std::memcpy(&value, &float_val, sizeof(value));
        break;
    }
    case element::f16: {
        const auto float16_val = ov::as_type_ptr<ov::op::v0::Constant>(n)->cast_vector<ov::float16>()[0];
        value = static_cast<int32_t>(float16_val.to_bits());
        break;
    }
    default: {
        OV_CPU_JIT_EMITTER_THROW("Doesn't support precision ", precision);
    }
    }
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
    auto dst_vreg = Xbyak_riscv::VReg(out[0]);

    // For now, use t0 as a temporary register
    Xbyak_riscv::Reg tmp_gpr = Xbyak_riscv::t0;

    // Load scalar value directly into register
    h->uni_li(tmp_gpr, value);

    // Move scalar from GPR to vector register and broadcast
    h->vmv_v_x(dst_vreg, tmp_gpr);
}

}  // namespace ov::intel_cpu::riscv64
