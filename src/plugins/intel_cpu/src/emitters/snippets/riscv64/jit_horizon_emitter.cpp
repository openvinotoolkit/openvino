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
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/horizon_max.hpp"
#include "snippets/op/horizon_sum.hpp"
#include "utils.hpp"
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

size_t jit_horizon_emitter::aux_gprs_count() const {
    return utils::get_snippet_lanes() <= 31 ? 0 : 1;
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
    const auto lane_count = ov::intel_cpu::riscv64::utils::get_snippet_lanes();

    auto src = Xbyak_riscv::VReg(in[0]);
    auto dst = Xbyak_riscv::VReg(out[0]);

    if (m_op_type == OpType::max) {
        h->vfredmax_vs(mask_vreg(), src, src);
    } else {
        h->vmv_v_x(mask_vreg(), Xbyak_riscv::zero);
        h->vfredosum_vs(mask_vreg(), src, mask_vreg());
    }

    set_vector_length(h, lane_count, Xbyak_riscv::SEW::e32, aux_gpr_idxs);
    h->vrgather_vi(dst, mask_vreg(), 0);
}

}  // namespace ov::intel_cpu::riscv64
