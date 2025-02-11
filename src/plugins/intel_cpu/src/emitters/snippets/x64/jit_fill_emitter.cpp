// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_fill_emitter.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

jit_fill_emitter::jit_fill_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                   dnnl::impl::cpu::x64::cpu_isa_t isa,
                                   const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa, ov::element::f32, emitter_in_out_map::vec_to_vec) {
    const auto fill = ov::as_type_ptr<snippets::op::Fill>(expr->get_node());
    if (fill->get_element_type().size() != 4) {
        OV_CPU_JIT_EMITTER_THROW("supports only 4 Byte element types but gets: ", fill->get_element_type());
    }

    offset = fill->get_offset();
    fill_value = fill->get_fill_value();
    if (!is_optimized()) {
        push_arg_entry_of("value", fill_value, true);
    }
    prepare_table();
}

size_t jit_fill_emitter::aux_gprs_count() const {
    // Optimized version (fill full vector by zero) doesn't need additional register
    if (is_optimized()) {
        return 0;
    }
    // + 1 reg for table value in full vector case
    if (is_full_reg()) {
        return 1;
    }
    // + 1 reg for temp reg for mask in avx512
    return one_of(host_isa_, dnnl::impl::cpu::x64::avx512_core) ? 2 : 1;
}

void jit_fill_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
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
void jit_fill_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    using Vmm = typename dnnl::impl::utils::
        conditional3<isa == dnnl::impl::cpu::x64::sse41, Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    auto src_vmm = Vmm(in[0]);
    auto dst_vmm = Vmm(out[0]);

    const size_t supported_et_size = 4;
    const auto register_capacity = (src_vmm.getBit() / 8) / supported_et_size;
    if (offset == register_capacity) {
        // WA: since AssignRegisters doesn't support inplace logic, Fill ops with offset = register_capacity can't be
        // removed from the LIR
        // TODO: when inplace is supported, remove such Fill ops from the LIR and remove this logic.
        // Ticket: 126270
        if (src_vmm.getIdx() != dst_vmm.getIdx()) {
            h->uni_vmovups(dst_vmm, src_vmm);
        }
    } else if (is_full_reg()) {
        fill_full<Vmm>(dst_vmm);
    } else {
        fill_tail<Vmm>(src_vmm, dst_vmm);
    }
}

template <typename Vmm>
void jit_fill_emitter::fill_full(const Vmm& dst_vmm) const {
    // Optimized impl for zero
    if (is_optimized()) {
        h->uni_vpxor(dst_vmm, dst_vmm, dst_vmm);
        return;
    }

    h->uni_vbroadcastss(dst_vmm, table_val("value"));
}

template <typename Vmm>
void jit_fill_emitter::fill_tail(const Vmm& src_vmm, const Vmm& dst_vmm) const {
    if (one_of(host_isa_, dnnl::impl::cpu::x64::avx512_core)) {
        uint64_t tail_mask = 1;
        tail_mask = ~((tail_mask << offset) - tail_mask);
        h->mov(Reg64(aux_gpr_idxs[0]), tail_mask);
        h->kmovq(k_mask, Reg64(aux_gpr_idxs[0]));
        h->vblendmps(dst_vmm | k_mask, src_vmm, table_val("value"));
    } else if (one_of(host_isa_, dnnl::impl::cpu::x64::avx2, dnnl::impl::cpu::x64::sse41)) {
        uint8 imm = 1;
        imm = ~((imm << offset) - imm);  // shift load_num bit
        if (host_isa_ == dnnl::impl::cpu::x64::sse41 && src_vmm.getIdx() != dst_vmm.getIdx()) {
            h->uni_vmovups(dst_vmm, src_vmm);
            h->uni_vblendps(dst_vmm, dst_vmm, table_val("value"), imm);
        } else {
            h->uni_vblendps(dst_vmm, src_vmm, table_val("value"), imm);
        }
    }
}

}  // namespace ov::intel_cpu
