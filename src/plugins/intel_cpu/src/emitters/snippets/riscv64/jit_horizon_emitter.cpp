// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_horizon_emitter.hpp"

#include <cstddef>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "emitters/utils.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/horizon_max.hpp"
#include "snippets/op/horizon_sum.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"
#include "xbyak_riscv/xbyak_riscv_csr.hpp"

namespace ov::intel_cpu::riscv64 {

using cpu_isa_t = ov::intel_cpu::riscv64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_horizon_emitter::jit_horizon_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                                         cpu_isa_t isa,
                                         const ExpressionPtr& expr)
    : jit_emitter(h, isa, ov::element::f32, emitter_in_out_map::vec_to_vec) {
    if (ov::is_type<const snippets::op::HorizonMax>(expr->get_node())) {
        m_op_type = OpType::max;
    } else if (ov::is_type<const snippets::op::HorizonSum>(expr->get_node())) {
        m_op_type = OpType::sum;
    } else {
        OV_CPU_JIT_EMITTER_THROW("Expects HorizonMax or HorizonSum ops");
    }
}

void jit_horizon_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (host_isa_ == ov::intel_cpu::riscv64::gv) {
        emit_isa<ov::intel_cpu::riscv64::gv>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_horizon_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    static constexpr size_t lane_count = 4;
    static constexpr auto stack_size = static_cast<int>(lane_count * sizeof(float));
    static constexpr auto elt_size = static_cast<int>(sizeof(float));

    OPENVINO_ASSERT(aux_gpr_idxs.size() >= 2, "Horizon emitter expects two auxiliary GPR registers");
    OPENVINO_ASSERT(aux_fp_gpr_idxs.size() >= 2, "Horizon emitter expects two auxiliary FP GPR registers");

    auto src = Xbyak_riscv::VReg(in[0]);
    auto dst = Xbyak_riscv::VReg(out[0]);
    auto active_lanes = Xbyak_riscv::Reg(static_cast<int>(aux_gpr_idxs[0]));
    auto iter = Xbyak_riscv::Reg(static_cast<int>(aux_gpr_idxs[1]));
    auto acc = Xbyak_riscv::FReg(static_cast<int>(aux_fp_gpr_idxs[0]));
    auto tmp = Xbyak_riscv::FReg(static_cast<int>(aux_fp_gpr_idxs[1]));
    Xbyak_riscv::Label reduce_done;
    Xbyak_riscv::Label reduce_loop;

    if (src.getIdx() != dst.getIdx()) {
        h->vmv_v_v(dst, src);
    }

    // Respect current active vector length (tail count) and reduce only active lanes.
    h->csrr(active_lanes, Xbyak_riscv::CSR::vl);
    h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, -stack_size);
    h->vse32_v(dst, Xbyak_riscv::sp);

    h->flw(acc, Xbyak_riscv::sp, 0);
    h->uni_li(iter, 1);
    h->bge(iter, active_lanes, reduce_done);
    h->addi(active_lanes, active_lanes, -1);
    h->addi(iter, Xbyak_riscv::sp, elt_size);
    h->L(reduce_loop);
    h->flw(tmp, iter, 0);
    if (m_op_type == OpType::max) {
        h->fmax_s(acc, acc, tmp);
    } else {
        h->fadd_s(acc, acc, tmp);
    }
    h->addi(iter, iter, elt_size);
    h->addi(active_lanes, active_lanes, -1);
    h->bne(active_lanes, Xbyak_riscv::zero, reduce_loop);
    h->L(reduce_done);

    // Keep x64 behavior: replicate reduced scalar to all lanes.
    h->vsetivli(Xbyak_riscv::zero, lane_count, Xbyak_riscv::SEW::e32, Xbyak_riscv::LMUL::m1);
    h->vfmv_v_f(dst, acc);
    h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, stack_size);
}

}  // namespace ov::intel_cpu::riscv64
