// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_horizon_emitter.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

jit_horizon_emitter::jit_horizon_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                         dnnl::impl::cpu::x64::cpu_isa_t isa,
                                         const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa, ov::element::f32, emitter_in_out_map::vec_to_vec) {
    if (ov::is_type<const snippets::op::HorizonMax>(expr->get_node())) {
        m_op_type = OpType::max;
    } else if (ov::is_type<const snippets::op::HorizonSum>(expr->get_node())) {
        m_op_type = OpType::sum;
    } else {
        OV_CPU_JIT_EMITTER_THROW("exprects HorizonMax or HorizonSum ops");
    }
}

void jit_horizon_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_horizon_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    auto src_vmm = Vmm(in[0]);
    auto dst_vmm = Vmm(out[0]);
    auto aux_vmm = Vmm(aux_vec_idxs[0]);

    if (in[0] != out[0]) {
        h->uni_vmovups(dst_vmm, src_vmm);
    }
    if (isa == dnnl::impl::cpu::x64::avx512_core) {
        auto dst_zmm = Xbyak::Zmm(out[0]);
        auto aux_zmm = Xbyak::Zmm(aux_vec_idxs[0]);
        h->vshuff32x4(aux_zmm, dst_zmm, dst_zmm, 0x4E);
        perform_op<Xbyak::Zmm>(dst_zmm, dst_zmm, aux_zmm);
        h->vshuff32x4(aux_zmm, dst_zmm, dst_zmm, 0xB1);
        perform_op<Xbyak::Zmm>(dst_zmm, dst_zmm, aux_zmm);
    } else if (isa == dnnl::impl::cpu::x64::avx2) {
        auto dst_ymm = Xbyak::Ymm(out[0]);
        auto aux_ymm = Xbyak::Ymm(aux_vec_idxs[0]);
        h->vperm2i128(aux_ymm, dst_ymm, dst_ymm, 0x01);
        perform_op<Xbyak::Ymm>(dst_ymm, dst_ymm, aux_ymm);
    }
    h->uni_vshufps(aux_vmm, dst_vmm, dst_vmm, 0x4E);
    perform_op<Xbyak::Xmm>(dst_vmm, dst_vmm, aux_vmm);
    h->uni_vshufps(aux_vmm, dst_vmm, dst_vmm, 0xB1);
    perform_op<Xbyak::Xmm>(dst_vmm, dst_vmm, aux_vmm);
}

template <typename Vmm>
void jit_horizon_emitter::perform_op(const Vmm& vmm1, const Vmm& vmm2, const Vmm& vmm3) const {
    switch (m_op_type) {
    case OpType::max:
        h->uni_vmaxps(vmm1, vmm2, vmm3);
        break;
    case OpType::sum:
        h->uni_vaddps(vmm1, vmm2, vmm3);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported horizontal operation.");
    }
}

}  // namespace ov::intel_cpu
