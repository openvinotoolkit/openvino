// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_fill_emitter.hpp"

#include "cpu/aarch64/xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h"
#include "emitters/utils.hpp"

using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_fill_emitter::jit_fill_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa, ov::element::f32, emitter_in_out_map::vec_to_vec) {
    const auto fill = ov::as_type_ptr<snippets::op::Fill>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(fill != nullptr, "Expects Fill expression");
    OV_CPU_JIT_EMITTER_ASSERT(fill->get_element_type().size() == 4,
                              "Supports only 4 Byte element types but gets ",
                              fill->get_element_type());

    offset = fill->get_offset();
    fill_value = fill->get_fill_value();
    if (!is_optimized()) {
        push_arg_entry_of("value", fill_value, true);
    }
    prepare_table();
}

size_t jit_fill_emitter::get_aux_gprs_count() const {
    // Optimized version (fill full vector by zero) doesn't need additional register
    if (is_optimized()) {
        return 0;
    }

    return 1;
}

void jit_fill_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Fill emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_fill_emitter::emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    if (is_full_reg()) {
        fill_full<isa>(out);
    } else {
        fill_tail<isa>(in, out);
    }
}

template <cpu_isa_t isa>
void jit_fill_emitter::fill_full(const std::vector<size_t>& out) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto dst = TReg(out[0]);

    // Optimized impl for zero
    if (is_optimized()) {
        h->uni_clear(dst);
        return;
    }

    AdrImm src = table_val("value");
    h->uni_ld1rw(dst.s, src.getXn(), src.getImm());
}

template <cpu_isa_t isa>
void jit_fill_emitter::fill_tail(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto dst = TReg(out[0]);

    switch (offset) {
    case 1:
        h->ld1(dst.s[1], table_val2("value", sizeof(float)));
        h->ld1(dst.d[1], table_val2("value", 2 * sizeof(float)));
        break;
    case 2:
        h->ld1(dst.d[1], table_val2("value", 2 * sizeof(float)));
        break;
    case 3:
        h->ld1(dst.s[3], table_val2("value", 3 * sizeof(float)));
        break;
    case 4:
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Fill emitter has unexpected offset ", offset);
    }
}

}  // namespace ov::intel_cpu::aarch64
